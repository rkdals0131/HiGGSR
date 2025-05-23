import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import threading
import time

# --- 시각화 함수들 ---
def visualize_density_map(density_map, x_edges, y_edges, cmap_name='viridis', vmin=None, vmax=None, title_suffix=""):
    if density_map.size == 0: print("시각화할 밀도 맵이 비어있습니다."); return
    plt.figure(figsize=(10, 8))
    extend_opt = 'neither'
    valid_data_exists = np.any(np.isfinite(density_map) & (density_map != 0))
    if valid_data_exists:
        finite_density_map = density_map[np.isfinite(density_map)]
        if finite_density_map.size > 0:
            if vmin is not None and vmax is not None:
                has_lower = np.any(finite_density_map < vmin); has_upper = np.any(finite_density_map > vmax)
                if has_lower and has_upper: extend_opt = 'both'
                elif has_lower: extend_opt = 'min'
                elif has_upper: extend_opt = 'max'
            elif vmin is not None and np.any(finite_density_map < vmin): extend_opt = 'min'
            elif vmax is not None and np.any(finite_density_map > vmax): extend_opt = 'max'
    
    img = plt.imshow(density_map.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap=cmap_name, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(img, label=f'Pillar Value ({title_suffix})', extend=extend_opt)
    plt.xlabel("X coordinate"); plt.ylabel("Y coordinate")
    vmin_str = f"{vmin:.2f}" if vmin is not None else 'auto'; vmax_str = f"{vmax:.2f}" if vmax is not None else 'auto'
    plt.title(f"2.5D Pillar Map ({title_suffix}, Color range: {vmin_str}-{vmax_str})")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    print("밀도 맵 시각화 창이 열렸습니다. 창을 닫으면 계속 진행됩니다.")
    try:
        plt.show()  # 창을 닫을 때까지 대기
    except Exception as e:
        print(f"시각화 오류: {e}")
    print("밀도 맵 시각화 창이 닫혔습니다. 계속 진행합니다.")

def visualize_2d_keypoint_registration(global_map_keypoints, scan_keypoints_original, scan_keypoints_transformed, global_map_x_edges, global_map_y_edges, title="2D Keypoint Registration Result"):
    plt.figure(figsize=(12, 10)) 
    if global_map_keypoints.shape[0] > 0: plt.scatter(global_map_keypoints[:, 0], global_map_keypoints[:, 1], c='blue', s=10, label='Global Map Keypoints', alpha=0.3)
    if scan_keypoints_transformed.shape[0] > 0: plt.scatter(scan_keypoints_transformed[:, 0], scan_keypoints_transformed[:, 1], c='red', s=30, marker='x', label='Live Scan Keypoints (Estimated)')
    plt.xlabel("X coordinate"); plt.ylabel("Y coordinate"); plt.title(title); plt.legend()
    if global_map_x_edges is not None and global_map_y_edges is not None: plt.xlim(global_map_x_edges[0], global_map_x_edges[-1]); plt.ylim(global_map_y_edges[0], global_map_y_edges[-1])
    plt.gca().set_aspect('equal', adjustable='box'); plt.grid(True, linestyle='--', alpha=0.5)
    
    print("키포인트 정합 시각화 창이 열렸습니다. 창을 닫으면 계속 진행됩니다.")
    try:
        plt.show()  # 창을 닫을 때까지 대기
    except Exception as e:
        print(f"시각화 오류: {e}")
    print("키포인트 정합 시각화 창이 닫혔습니다. 계속 진행합니다.")

def visualize_3d_registration_o3d(global_map_points_3d_np, live_scan_points_3d_np, final_transform_4x4, global_map_color=[0.7,0.7,0.7], live_scan_color=[1,0,0]):
    if global_map_points_3d_np.shape[0] == 0 or live_scan_points_3d_np.shape[0] == 0: 
        print("3D 시각화를 위한 포인트가 부족합니다."); 
        return
    
    pcd_global = o3d.geometry.PointCloud(); 
    pcd_global.points = o3d.utility.Vector3dVector(global_map_points_3d_np); 
    pcd_global.paint_uniform_color(global_map_color)
    
    pcd_scan = o3d.geometry.PointCloud(); 
    pcd_scan.points = o3d.utility.Vector3dVector(live_scan_points_3d_np); 
    pcd_scan.transform(final_transform_4x4); 
    pcd_scan.paint_uniform_color(live_scan_color)
    
    print("3D 시각화 창이 열립니다. 창을 닫으면 계속 진행됩니다.")
    try:
        o3d.visualization.draw_geometries([pcd_global, pcd_scan], 
                                        window_name="3D Global Registration Result", 
                                        width=1024, height=768)
    except Exception as e:
        print(f"3D 시각화 오류: {e}")
    print("3D 시각화 창이 닫혔습니다. 계속 진행합니다.")

def visualize_super_grid_scores(
    pillar_map_data, 
    pillar_map_x_edges, 
    pillar_map_y_edges, 
    all_raw_cell_infos_this_level, 
    search_area_details_list_this_level, 
    global_map_x_edges, 
    global_map_y_edges, 
    current_level_selected_candidates, 
    next_level_config, 
    initial_map_edges_for_relative_calc, 
    title_suffix=""
):
    valid_infos_for_scatter = [
        info for info in all_raw_cell_infos_this_level 
        if info[0] is not None and np.isfinite(info[4]) and np.isfinite(info[5])
    ]

    score_map = {(info[4], info[5]): info[0] for info in valid_infos_for_scatter}
    unique_scores = sorted(list(set(info[0] for info in valid_infos_for_scatter)))
    
    if not valid_infos_for_scatter and not current_level_selected_candidates:
        print(f"Level '{title_suffix}': 표시할 셀 점수 데이터도 없고, 선택된 후보도 없어 시각화를 건너뜁니다.")
        return

    plt.figure(figsize=(13, 11))
    ax = plt.gca() 
    
    if pillar_map_data is not None and pillar_map_data.size > 0 and \
       pillar_map_x_edges is not None and pillar_map_y_edges is not None:
        ax.imshow(pillar_map_data.T, origin='lower', 
                   extent=[pillar_map_x_edges[0], pillar_map_x_edges[-1], pillar_map_y_edges[0], pillar_map_y_edges[-1]],
                   cmap='gray', aspect='auto', alpha=0.3, zorder=0) 

    norm = plt.Normalize(vmin=min(unique_scores) if unique_scores else 0, vmax=max(unique_scores) if unique_scores else 1)
    cmap = plt.cm.viridis
    
    cells_drawn_for_legend = False
    if search_area_details_list_this_level:
        for area_detail in search_area_details_list_this_level:
            area_x_edges, area_y_edges = area_detail["bounds"]
            super_x_edges, super_y_edges = area_detail["grid_edges"]
            
            if super_x_edges is None or super_y_edges is None or len(super_x_edges) <=1 or len(super_y_edges) <=1:
                continue

            for i in range(len(super_x_edges) - 1):
                for j in range(len(super_y_edges) - 1):
                    cell_x_min, cell_x_max = super_x_edges[i], super_x_edges[i+1]
                    cell_y_min, cell_y_max = super_y_edges[j], super_y_edges[j+1]
                    
                    cell_cx = (cell_x_min + cell_x_max) / 2
                    cell_cy = (cell_y_min + cell_y_max) / 2
                    
                    current_cell_score = None
                    tolerance = 1e-6 
                    for score_cx, score_cy in score_map.keys():
                        if abs(score_cx - cell_cx) < tolerance and abs(score_cy - cell_cy) < tolerance:
                            current_cell_score = score_map[(score_cx, score_cy)]
                            break
                    
                    if current_cell_score is not None:
                        cell_color = cmap(norm(current_cell_score))
                        rect = plt.Rectangle((cell_x_min, cell_y_min), 
                                             cell_x_max - cell_x_min, 
                                             cell_y_max - cell_y_min,
                                             facecolor=cell_color, alpha=0.7, edgecolor='none',
                                             zorder=1, label="Scored Cell" if not cells_drawn_for_legend else None)
                        ax.add_patch(rect)
                        if not cells_drawn_for_legend: cells_drawn_for_legend = True
    
    if cells_drawn_for_legend:
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([]) 
        plt.colorbar(mappable, ax=ax, label="Cell Score", fraction=0.046, pad=0.04)

    if search_area_details_list_this_level:
        for i, area_detail in enumerate(search_area_details_list_this_level):
            x_e, y_e = area_detail["bounds"] 
            if x_e is not None and y_e is not None and len(x_e)==2 and len(y_e)==2:
                 rect_w = x_e[1] - x_e[0]
                 rect_h = y_e[1] - y_e[0]
                 if rect_w > 0 and rect_h > 0:
                    search_rect = plt.Rectangle((x_e[0], y_e[0]), rect_w, rect_h,
                                                linewidth=1.5, edgecolor='orange', facecolor='none', linestyle=':', alpha=0.8, 
                                                label="Searched Area (Overall)" if i == 0 else None, zorder=2)
                    ax.add_patch(search_rect)

    if current_level_selected_candidates:
        num_cand_to_plot = min(len(current_level_selected_candidates), 5) 
        for i, cand_info in enumerate(current_level_selected_candidates[:num_cand_to_plot]):
            cand_score, cand_tx, cand_ty, cand_theta, cand_cx, cand_cy = cand_info
            plot_tx, plot_ty = cand_tx, cand_ty 

            marker_style = 'r*' if i == 0 else ('b*' if i == 1 else ('g*' if i == 2 else 'c*'))
            label_text = f'Cand{i+1} (S:{cand_score:.0f} P:{plot_tx:.1f},{plot_ty:.1f})'
            ax.plot(plot_tx, plot_ty, marker_style, markersize=15, label=label_text, alpha=0.95, markeredgecolor='k', zorder=4) 

            if next_level_config: 
                area_type_next = next_level_config.get("search_area_type")
                area_param_next = next_level_config.get("area_ratio_or_size")
                center_x_next, center_y_next = cand_tx, cand_ty 
                search_w_next, search_h_next = 0, 0

                if area_type_next == "relative_to_map" and initial_map_edges_for_relative_calc and area_param_next is not None:
                    init_map_x_edges_calc, init_map_y_edges_calc = initial_map_edges_for_relative_calc
                    map_w_init = init_map_x_edges_calc[-1] - init_map_x_edges_calc[0]
                    map_h_init = init_map_y_edges_calc[-1] - init_map_y_edges_calc[0]
                    ratio = area_param_next
                    search_w_next = map_w_init * np.sqrt(ratio)
                    search_h_next = map_h_init * np.sqrt(ratio)
                elif area_type_next == "absolute_size" and area_param_next is not None:
                    search_w_next, search_h_next = area_param_next 
                
                if search_w_next > 0 and search_h_next > 0:
                    rect_x_next = center_x_next - search_w_next / 2
                    rect_y_next = center_y_next - search_h_next / 2
                    next_area_label = f'Next Area (Cand {i+1})' if i < 2 else None 
                    next_rect = plt.Rectangle((rect_x_next, rect_y_next), search_w_next, search_h_next, 
                                         linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--', alpha=0.85, label=next_area_label, zorder=3) 
                    ax.add_patch(next_rect)
    
    ax.set_xlabel("X coordinate (Global Map)")
    ax.set_ylabel("Y coordinate (Global Map)")
    ax.set_title(f"Hierarchical Search Heatmap ({title_suffix})")
    
    if global_map_x_edges is not None and len(global_map_x_edges) > 1:
        ax.set_xlim(global_map_x_edges[0], global_map_x_edges[-1])
    if global_map_y_edges is not None and len(global_map_y_edges) > 1:
        ax.set_ylim(global_map_y_edges[0], global_map_y_edges[-1])
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0.5) 
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label: 
      ax.legend(by_label.values(), by_label.keys(), fontsize='small', loc='upper right') 
    
    print(f"히트맵 시각화 (레벨 {title_suffix}) 창이 열렸습니다. 창을 닫으면 계속 진행됩니다.")
    try:
        plt.show()  # 창을 닫을 때까지 대기
    except Exception as e:
        print(f"시각화 오류: {e}")
    print(f"히트맵 시각화 (레벨 {title_suffix}) 창이 닫혔습니다. 계속 진행합니다.") 