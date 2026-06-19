import numpy as np
import tifffile

import background_substraction as bg
import dualview_register as dvr
import hot_pixel_remover as hpr
import line_mask as lm

fd_h = r"C:\Users\Ruiz\Desktop\20260519115441_line_scanning_beads500nm_h.tif"
fd_v = r"C:\Users\Ruiz\Desktop\20260519115953_line_scanning_beads500nm_v.tif"
img_stack_h = tifffile.imread(fd_h)
img_stack_v = tifffile.imread(fd_v)

pixel_size = 162.5  # nm
max_detection_frames = 310

hot_mask_h, qc_h = hpr.detect_hot_pixels_from_stack(img_stack_h,
                                                    max_detection_frames=max_detection_frames,
                                                    spatial_size=3,
                                                    spatial_k=4,
                                                    temporal_k=4,
                                                    persistence_fraction=0.4,
                                                    max_component_size=4)
img_stack_h_corrected = hpr.replace_hot_pixels(img_stack_h, hot_mask_h)

hot_mask_v, qc_v = hpr.detect_hot_pixels_from_stack(img_stack_v,
                                                    max_detection_frames=max_detection_frames,
                                                    spatial_size=3,
                                                    spatial_k=4,
                                                    temporal_k=4,
                                                    persistence_fraction=0.4,
                                                    max_component_size=4)
img_stack_v_corrected = hpr.replace_hot_pixels(img_stack_v, hot_mask_v)

n_dark = 5
dark_h = img_stack_h_corrected[:n_dark]
data_h = img_stack_h_corrected[n_dark:]
dark_v = img_stack_v_corrected[:n_dark]
data_v = img_stack_v_corrected[n_dark:]

img = np.concatenate((data_h, data_v), axis=0).max(axis=0)
top = img[: img.shape[0] // 2, :]
bottom = img[img.shape[0] // 2:, :]
result_rg = dvr.register_dualview(top, bottom,
                                  model="similarity",
                                  refine="polynomial",
                                  build_index_map=True,
                                  pixel_size_nm=pixel_size)
top_stack_h = data_h[:, : hot_mask_h.shape[0] // 2, :]
bottom_stack_h = data_h[:, hot_mask_h.shape[0] // 2:, :]
bot_stack_h = np.zeros(bottom_stack_h.shape)
for i in range(top_stack_h.shape[0]):
    bot_stack_h[i] = dvr.gather_nearest(bottom_stack_h[i], result_rg.idx_x, result_rg.idx_y, result_rg.valid)
    top_stack_h[i] *= result_rg.valid

top_stack_v = data_v[:, : hot_mask_v.shape[0] // 2, :]
bottom_stack_v = data_v[:, hot_mask_v.shape[0] // 2:, :]
bot_stack_v = np.zeros(bottom_stack_v.shape)
for i in range(top_stack_v.shape[0]):
    bot_stack_v[i] = dvr.gather_nearest(bottom_stack_v[i], result_rg.idx_x, result_rg.idx_y, result_rg.valid)
    top_stack_v[i] *= result_rg.valid

mask_stack_h = np.zeros_like(top_stack_h)
for i in range(top_stack_h.shape[0]):
    msk_v, info_v = lm.extract_line_mask(top_stack_h[i], line_width=5, continuous=False)
    msk_h, info_h = lm.extract_line_mask(bot_stack_h[i], line_width=5, continuous=False)
    mask_stack_h[i] = msk_v & msk_h

mask_stack_v = np.zeros_like(top_stack_v)
for i in range(top_stack_v.shape[0]):
    msk_v, info_v = lm.extract_line_mask(top_stack_v[i], line_width=5, continuous=False)
    msk_h, info_h = lm.extract_line_mask(bot_stack_v[i], line_width=5, continuous=False)
    mask_stack_v[i] = msk_v & msk_h

result_bg_h_top = bg.estimate_background(top_stack_h,
                                   n_dark=0,
                                   method="median",
                                   n_bins=1,
                                   gain_e_per_adu=None)
# bg_sub_h_top = bg.subtract_background(top_stack_h, result_bg_h_top)
signal_h_top, signal_mask_h_top = bg.extract_signal(top_stack_h, result_bg_h_top, k=4.0)

result_bg_h_bot = bg.estimate_background(bot_stack_h,
                                   n_dark=0,
                                   method="median",
                                   n_bins=1,
                                   gain_e_per_adu=None)
# bg_sub_h_bot = bg.subtract_background(bot_stack_h, result_bg_h_bot)
signal_h_bot, signal_mask_h_bot = bg.extract_signal(bot_stack_h, result_bg_h_bot, k=4.0)

result_bg_v_top = bg.estimate_background(top_stack_v,
                                   n_dark=0,
                                   method="median",
                                   n_bins=1,
                                   gain_e_per_adu=None)
# bg_sub_h_top = bg.subtract_background(top_stack_v, result_bg_h_top)
signal_v_top, signal_mask_v_top = bg.extract_signal(top_stack_v, result_bg_v_top, k=4.0)

result_bg_h_top = bg.estimate_background(top_stack_v,
                                   n_dark=0,
                                   method="median",
                                   n_bins=1,
                                   gain_e_per_adu=None)
# bg_sub_h_top = bg.subtract_background(top_stack_v, result_bg_h_top)
signal_h_top, signal_mask_h_top = bg.extract_signal(top_stack_v, result_bg_v_top, k=4.0)


top_stack_h *= mask_stack_h
bot_stack_h *= mask_stack_h

top_stack_v *= mask_stack_v
bot_stack_v *= mask_stack_v

G = np.sqrt(bot_stack_h * bot_stack_v / (top_stack_h * top_stack_v))
K2 = np.sqrt(top_stack_v * bot_stack_v / (top_stack_h * bot_stack_h))
il_m = top_stack_h + bot_stack_v / G / K2
ip_m = bot_stack_h / G + top_stack_v / K2
r_m = (il_m - ip_m) / (il_m + 2 * ip_m)

tifffile.imwrite(r"C:\Users\Ruiz\Desktop\20260519115441_line_scanning_beads500nm_anistropy.tif", r_m)


# ihh, ihv = top_stack_h, bot_stack_h
# ivh, ivv = top_stack_v, bot_stack_v

# ihh_m = np.mean(ihh)
# std_hh = np.std(ihh)
# ihv_m = np.mean(ihv)
# std_hv = np.std(ihv)
# ivh_m = np.mean(ivh)
# std_vh = np.std(ivh)
# ivv_m = np.mean(ivv)

# G = np.sqrt(ihv_m * ivv_m / (ihh_m * ivh_m))
# K2 = np.sqrt(ivh_m * ivv_m / (ihh_m * ihv_m))

# il_m = ihh_m + ivv_m / G / K2
# ip_m = ihv_m / G + ivh_m / K2
# r_m = (il_m - ip_m) / (il_m + 2 * ip_m)
# dr_m = (((1 - r_m) ** 2) * (1 + 2 * r_m) * (1 - r_m + G * (1 + 2 * r_m))) / (3 * (il_m + 2 * G * ip_m))

# x_n = np.sqrt((ihv_m * ivh_m) / (ihh_m * ivv_m))
# r_n = (1 - x_n) / (1 + 2 * x_n)
# rs_n = (3 * x_n) / (1 + x_n) ** 2 / 2 * np.sqrt((std_hh / ihh_m) ** 2 + (std_hv / ihv_m) ** 2 + (std_vh / ivh_m) ** 2 + (std_vv / ivv_m ** 2))
