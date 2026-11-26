from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyvista as pv
from pyvistaqt import QtInteractor

from src.viz.config import DEFAULT_ELECTRODES, get_ordered_roi_names

SCENE_BG_COLOR = "#0b0f17"
CORTEX_NEUTRAL_COLOR = "#94a3b8"
SCALP_COLOR = "#cbd5e1"
ELECTRODE_COLOR = "#f59e0b"
HIGHLIGHT_COLOR = "#38bdf8"


def _get_standard_montage() -> mne.channels.DigMontage:
    try:
        return mne.channels.make_standard_montage("colin27_1005")
    except ValueError:
        return mne.channels.make_standard_montage("standard_1005")


class BrainView3DWidget(QWidget):
    """Интерактивная 3D-сцена головного мозга fsaverage + скальп + электроды 10–20."""

    roi_clicked = Signal(int, str)

    def __init__(
        self,
        subjects_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._subjects_dir = Path(subjects_dir) if subjects_dir is not None else None
        self._roi_names: list[str] = get_ordered_roi_names()

        # Геометрия поверхности коры и соответствие вершин 68 зонам
        self._lh_verts: np.ndarray | None = None
        self._rh_verts: np.ndarray | None = None
        self._lh_mesh: pv.PolyData | None = None
        self._rh_mesh: pv.PolyData | None = None
        self._lh_v2roi: np.ndarray | None = None
        self._rh_v2roi: np.ndarray | None = None
        self._roi_centroids: np.ndarray | None = None

        # Предвыделенные буферы скаляров (обновление без реаллокаций)
        self._lh_mask: np.ndarray | None = None
        self._rh_mask: np.ndarray | None = None
        self._lh_masked_rois: np.ndarray | None = None
        self._rh_masked_rois: np.ndarray | None = None
        self._lh_scalars: np.ndarray | None = None
        self._rh_scalars: np.ndarray | None = None

        # Актёры PyVista
        self._lh_actor = None
        self._rh_actor = None
        self._scalp_actor = None
        self._elec_actor = None
        self._elec_labels_actor = None
        self._highlight_actor = None

        # Параметры отображения
        self._cmap: str = "inferno"
        self._threshold_ratio: float = 0.20
        self._last_values: np.ndarray = np.zeros(len(self._roi_names), dtype=np.float32)
        self._last_vmax: float = 1.0
        self._scalp_visible: bool = True
        self._scalp_opacity: float = 0.15
        self._electrodes_visible: bool = True
        self._elec_labels_visible: bool = True
        self._lh_visible: bool = True
        self._rh_visible: bool = True
        self._selected_roi_idx: int | None = None

        # Кэш координат электродов для перерисовки подписей после инициализации OpenGL
        self._last_elec_points: np.ndarray | None = None
        self._last_elec_names: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background(SCENE_BG_COLOR)
        try:
            self.plotter.ren_win.SetMultiSamples(0)
        except Exception:
            pass
        layout.addWidget(self.plotter.interactor)

        self._init_scene()
        QTimer.singleShot(120, self._refresh_labels_after_init)

    @property
    def roi_names(self) -> list[str]:
        return list(self._roi_names)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(50, self._refresh_labels_after_init)

    def _refresh_labels_after_init(self) -> None:
        if self._elec_labels_visible and self._last_elec_points is not None:
            self._rebuild_electrode_labels()
            self.plotter.render()

    # ------------------------------------------------------------------
    # Инициализация 3D-модели fsaverage (кора, скальп, электроды)
    # ------------------------------------------------------------------

    def _resolve_fsaverage_dir(self) -> Path:
        if self._subjects_dir is not None and (self._subjects_dir / "fsaverage").exists():
            return self._subjects_dir / "fsaverage"
        fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
        return fs_dir

    @staticmethod
    def _faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
        n = faces.shape[0]
        return np.hstack([np.full((n, 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()

    def _init_scene(self) -> None:
        fs_dir = self._resolve_fsaverage_dir()
        subjects_root = fs_dir.parent

        # 1. Загружаем поверхности коры pial (в миллиметрах)
        lh_verts, lh_faces = mne.read_surface(fs_dir / "surf" / "lh.pial")
        rh_verts, rh_faces = mne.read_surface(fs_dir / "surf" / "rh.pial")
        self._lh_verts = np.asarray(lh_verts, dtype=np.float32)
        self._rh_verts = np.asarray(rh_verts, dtype=np.float32)

        self._lh_mesh = pv.PolyData(self._lh_verts, self._faces_to_pyvista(lh_faces))
        self._rh_mesh = pv.PolyData(self._rh_verts, self._faces_to_pyvista(rh_faces))

        # 2. Загружаем парцелляцию Desikan-Killiany (aparc) -> 68 зон
        labels = mne.read_labels_from_annot(
            "fsaverage",
            parc="aparc",
            subjects_dir=subjects_root,
            verbose=False,
        )
        name_to_idx = {name: i for i, name in enumerate(self._roi_names)}

        self._lh_v2roi = np.full(len(self._lh_verts), -1, dtype=np.int32)
        self._rh_v2roi = np.full(len(self._rh_verts), -1, dtype=np.int32)
        centroids = np.zeros((len(self._roi_names), 3), dtype=np.float32)

        for lbl in labels:
            idx = name_to_idx.get(lbl.name)
            if idx is None:
                continue
            verts_idx = np.asarray(lbl.vertices, dtype=np.int32)
            if lbl.hemi == "lh":
                valid = verts_idx[verts_idx < len(self._lh_verts)]
                self._lh_v2roi[valid] = idx
                if len(valid) > 0:
                    centroids[idx] = self._lh_verts[valid].mean(axis=0)
            elif lbl.hemi == "rh":
                valid = verts_idx[verts_idx < len(self._rh_verts)]
                self._rh_v2roi[valid] = idx
                if len(valid) > 0:
                    centroids[idx] = self._rh_verts[valid].mean(axis=0)

        self._roi_centroids = centroids
        self._lh_mask = self._lh_v2roi >= 0
        self._rh_mask = self._rh_v2roi >= 0
        self._lh_masked_rois = self._lh_v2roi[self._lh_mask]
        self._rh_masked_rois = self._rh_v2roi[self._rh_mask]
        self._lh_scalars = np.full(len(self._lh_verts), -1.0, dtype=np.float32)
        self._rh_scalars = np.full(len(self._rh_verts), -1.0, dtype=np.float32)

        self._lh_mesh.point_data["activity"] = self._lh_scalars
        self._rh_mesh.point_data["activity"] = self._rh_scalars

        scalar_bar_args = {
            "title": "sLORETA",
            "color": "#e2e8f0",
            "title_font_size": 12,
            "label_font_size": 10,
            "n_labels": 4,
            "fmt": "%.2f",
            "position_x": 0.30,
            "position_y": 0.03,
            "width": 0.40,
            "height": 0.045,
        }

        self._lh_actor = self.plotter.add_mesh(
            self._lh_mesh,
            scalars="activity",
            cmap=self._cmap,
            clim=[0.2, 1.0],
            below_color=CORTEX_NEUTRAL_COLOR,
            nan_color=CORTEX_NEUTRAL_COLOR,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args=scalar_bar_args,
            name="cortex_lh",
        )
        self._rh_actor = self.plotter.add_mesh(
            self._rh_mesh,
            scalars="activity",
            cmap=self._cmap,
            clim=[0.2, 1.0],
            below_color=CORTEX_NEUTRAL_COLOR,
            nan_color=CORTEX_NEUTRAL_COLOR,
            smooth_shading=True,
            show_scalar_bar=False,
            name="cortex_rh",
        )

        # 3. Полупрозрачный скальп головы
        self._load_scalp_mesh(fs_dir)

        # 4. Электроды 10–20 и подписи по умолчанию
        self.update_electrodes(DEFAULT_ELECTRODES)

        # 5. Правый клик по коре выбирает корковую зону
        try:
            self.plotter.enable_point_picking(
                callback=self._on_surface_picked,
                show_message=False,
                show_point=False,
                left_clicking=False,
                use_picker=True,
            )
        except Exception:
            pass

        self.set_camera_preset("front")

    def _load_scalp_mesh(self, fs_dir: Path) -> None:
        scalp_mesh: pv.PolyData | None = None
        bem_head = fs_dir / "bem" / "fsaverage-head.fif"
        outer_skin = fs_dir / "bem" / "outer_skin.surf"

        if bem_head.exists():
            surfs = mne.read_bem_surfaces(bem_head, verbose=False)
            if surfs:
                rr = np.asarray(surfs[0]["rr"], dtype=np.float32) * 1000.0
                tris = np.asarray(surfs[0]["tris"], dtype=np.int64)
                scalp_mesh = pv.PolyData(rr, self._faces_to_pyvista(tris))
        elif outer_skin.exists():
            rr, tris = mne.read_surface(outer_skin)
            scalp_mesh = pv.PolyData(
                np.asarray(rr, dtype=np.float32),
                self._faces_to_pyvista(tris),
            )

        if scalp_mesh is not None:
            self._scalp_actor = self.plotter.add_mesh(
                scalp_mesh,
                color=SCALP_COLOR,
                opacity=self._scalp_opacity,
                smooth_shading=True,
                pickable=False,
                name="scalp_head",
            )
            self._scalp_actor.SetVisibility(self._scalp_visible)

    # ------------------------------------------------------------------
    # Обновление активности, электродов и подписей
    # ------------------------------------------------------------------

    def update_activity(
        self,
        roi_values: np.ndarray,
        vmax: float | None = None,
    ) -> None:
        """Обновляет тепловую карту активности по 68 корковым зонам."""
        if self._lh_mesh is None or self._rh_mesh is None:
            return

        vals = np.asarray(roi_values, dtype=np.float32).ravel()
        if vals.shape[0] != len(self._roi_names):
            raise ValueError(
                f"Ожидалось {len(self._roi_names)} значений ROI, получено {vals.shape[0]}."
            )

        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_values = vals.copy()

        if vmax is None or not np.isfinite(vmax) or vmax <= 1e-9:
            peak = float(np.max(vals))
            self._last_vmax = peak if peak > 1e-9 else 1.0
        else:
            self._last_vmax = float(vmax)

        self._lh_scalars[self._lh_mask] = vals[self._lh_masked_rois]
        self._rh_scalars[self._rh_mask] = vals[self._rh_masked_rois]

        self._lh_mesh.point_data["activity"] = self._lh_scalars
        self._rh_mesh.point_data["activity"] = self._rh_scalars

        self._apply_scalar_range()
        self.plotter.render()

    def _apply_scalar_range(self) -> None:
        vmax = max(1e-6, self._last_vmax)
        vmin = max(1e-7, vmax * self._threshold_ratio)
        if vmin >= vmax:
            vmin = vmax * 0.95
        for actor in (self._lh_actor, self._rh_actor):
            if actor is not None:
                actor.mapper.scalar_range = (vmin, vmax)

    def update_electrodes(
        self,
        ch_names: list[str],
        montage: mne.channels.DigMontage | None = None,
    ) -> None:
        """Размещает 3D-сферы электродов и текстовые подписи каналов на поверхности скальпа."""
        std_montage = _get_standard_montage()
        std_pos = {
            k.lower(): np.asarray(v, dtype=np.float32) * 1000.0
            for k, v in std_montage.get_positions()["ch_pos"].items()
        }

        custom_pos: dict[str, np.ndarray] = {}
        if montage is not None:
            for k, v in montage.get_positions()["ch_pos"].items():
                arr = np.asarray(v, dtype=np.float32)
                if np.all(np.isfinite(arr)) and np.linalg.norm(arr) > 1e-4:
                    custom_pos[k.lower()] = arr * 1000.0

        points: list[np.ndarray] = []
        valid_names: list[str] = []
        for ch in ch_names:
            key = ch.strip().lower()
            pos = custom_pos.get(key)
            if pos is None:
                pos = std_pos.get(key)
            if pos is not None:
                points.append(pos)
                valid_names.append(ch.strip())

        if self._elec_actor is not None:
            self.plotter.remove_actor(self._elec_actor, render=False)
            self._elec_actor = None

        if not points:
            self._last_elec_points = None
            self._last_elec_names = []
            self._rebuild_electrode_labels()
            self.plotter.render()
            return

        pts = np.vstack(points).astype(np.float32)
        self._last_elec_points = pts
        self._last_elec_names = valid_names

        cloud = pv.PolyData(pts)
        self._elec_actor = self.plotter.add_mesh(
            cloud,
            color=ELECTRODE_COLOR,
            point_size=13,
            render_points_as_spheres=True,
            pickable=False,
            name="eeg_electrodes",
        )
        self._elec_actor.SetVisibility(self._electrodes_visible)

        self._rebuild_electrode_labels()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _rebuild_electrode_labels(self) -> None:
        if self._elec_labels_actor is not None:
            self.plotter.remove_actor(self._elec_labels_actor, render=False)
            self._elec_labels_actor = None

        if (
            not self._elec_labels_visible
            or self._last_elec_points is None
            or len(self._last_elec_names) == 0
        ):
            return

        # Выносим якоря подписей на 6 мм наружу от центра головы, чтобы скальп не перекрывал текст
        norms = np.linalg.norm(self._last_elec_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        label_pts = self._last_elec_points + (self._last_elec_points / norms) * 6.0

        self._elec_labels_actor = self.plotter.add_point_labels(
            label_pts,
            self._last_elec_names,
            font_size=11,
            text_color="#f8fafc",
            point_color=ELECTRODE_COLOR,
            point_size=4,
            render_points_as_spheres=False,
            shape="rounded_rect",
            shape_color="#0f172a",
            shape_opacity=0.72,
            margin=2,
            always_visible=True,
            bold=True,
            name="eeg_electrode_labels",
        )
        if self._elec_labels_actor is not None:
            self._elec_labels_actor.SetVisibility(self._elec_labels_visible)
        self.plotter.reset_camera_clipping_range()

    # ------------------------------------------------------------------
    # Управление слоями, палитрой, порогом и камерой
    # ------------------------------------------------------------------

    def set_scalp_visible(self, visible: bool) -> None:
        self._scalp_visible = bool(visible)
        if self._scalp_actor is not None:
            self._scalp_actor.SetVisibility(self._scalp_visible)
            self.plotter.render()

    def set_scalp_opacity(self, opacity: float) -> None:
        self._scalp_opacity = float(np.clip(opacity, 0.0, 1.0))
        if self._scalp_actor is not None:
            self._scalp_actor.GetProperty().SetOpacity(self._scalp_opacity)
            self.plotter.render()

    def set_electrodes_visible(self, visible: bool) -> None:
        self._electrodes_visible = bool(visible)
        if self._elec_actor is not None:
            self._elec_actor.SetVisibility(self._electrodes_visible)
            self.plotter.render()

    def set_electrode_labels_visible(self, visible: bool) -> None:
        self._elec_labels_visible = bool(visible)
        if self._elec_labels_visible:
            self._rebuild_electrode_labels()
        elif self._elec_labels_actor is not None:
            self._elec_labels_actor.SetVisibility(False)
        self.plotter.render()

    def set_hemispheres_visible(self, left: bool, right: bool) -> None:
        self._lh_visible = bool(left)
        self._rh_visible = bool(right)
        if self._lh_actor is not None:
            self._lh_actor.SetVisibility(self._lh_visible)
        if self._rh_actor is not None:
            self._rh_actor.SetVisibility(self._rh_visible)
        self.plotter.render()

    def set_colormap(self, cmap_name: str) -> None:
        self._cmap = cmap_name
        for actor in (self._lh_actor, self._rh_actor):
            if actor is not None:
                actor.mapper.lookup_table.cmap = cmap_name
        self._apply_scalar_range()
        self.plotter.render()

    def set_threshold(self, threshold_ratio: float) -> None:
        self._threshold_ratio = float(np.clip(threshold_ratio, 0.0, 0.90))
        self._apply_scalar_range()
        self.plotter.render()

    def set_camera_preset(self, preset: str) -> None:
        dist = 360.0
        views = {
            "front": ((0.0, dist, 20.0), (0.0, 0.0, 10.0), (0.0, 0.0, 1.0)),
            "back": ((0.0, -dist, 20.0), (0.0, 0.0, 10.0), (0.0, 0.0, 1.0)),
            "top": ((0.0, -10.0, dist), (0.0, 0.0, 10.0), (0.0, 1.0, 0.0)),
            "left": ((-dist, 0.0, 15.0), (0.0, 0.0, 10.0), (0.0, 0.0, 1.0)),
            "right": ((dist, 0.0, 15.0), (0.0, 0.0, 10.0), (0.0, 0.0, 1.0)),
        }
        cam = views.get(preset, views["front"])
        self.plotter.camera_position = cam
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def highlight_roi(self, roi_idx: int | None) -> None:
        if self._highlight_actor is not None:
            self.plotter.remove_actor(self._highlight_actor, render=False)
            self._highlight_actor = None

        self._selected_roi_idx = roi_idx
        if (
            roi_idx is not None
            and self._roi_centroids is not None
            and 0 <= roi_idx < len(self._roi_centroids)
        ):
            center = self._roi_centroids[roi_idx]
            if np.linalg.norm(center) > 1e-3:
                sphere = pv.Sphere(radius=7.5, center=center)
                self._highlight_actor = self.plotter.add_mesh(
                    sphere,
                    color=HIGHLIGHT_COLOR,
                    opacity=0.85,
                    pickable=False,
                    name="selected_roi_marker",
                )
        self.plotter.render()

    def _on_surface_picked(self, picked_point) -> None:
        if picked_point is None or self._roi_centroids is None:
            return
        pt = np.asarray(picked_point, dtype=np.float32)
        dists = np.linalg.norm(self._roi_centroids - pt, axis=1)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] < 45.0:
            self.highlight_roi(nearest_idx)
            self.roi_clicked.emit(nearest_idx, self._roi_names[nearest_idx])

    def cleanup(self) -> None:
        try:
            self.plotter.close()
        except Exception:
            pass