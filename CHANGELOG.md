# 更新日誌

## v1.1.1
- 2025 年 10 月 14 日

### 🐛 問題修復
- 將 `.bat` 與 `.sh` 腳本中的完成與錯誤提示訊息移至 `.py` 檔案，避免在部分終端機環境下執行時出現錯誤。

## v1.1.0
- 2025 年 10 月 11 日

### ✨ 新增功能
- 新增 VQ-VAE 與 LDM 模型的接續訓練功能（透過 `RESUME` 腳本參數啟用）。
- 新增 VQ-VAE 與 LDM 模型的混合精度訓練功能（透過 `USE_AMP` 腳本參數啟用）。
- 新增 VQ-VAE 訓練階段的即時字形樣本生成功能（透過 `IMG_SAVE_INTERVAL` 決定生成樣本儲存週期間隔）。
- 支援最新版本的 [Jigmo](https://kamichikoichi.github.io/jigmo/) 字型（該版本已完整涵蓋 [Unicode 17.0](https://www.unicode.org/charts/PDF/Unicode-17.0/) 漢字字集）。
- 新增 Windows 平台的專用腳本（`.bat` 格式腳本）與相關使用說明。

### 🔧 改進項目
- 更新 Unihan 字集至 Unicode 17.0（包含 `ext_c.txt`、`ext_e.txt`、`ext_j.txt` 等 Unihan 字集文件）。
- 改進 `prepare_dataset` 腳本，使其具備自動清理 `data/` 資料夾的功能。
- 改進腳本執行時的終端機提示訊息。
- 調整字形生成樣本的檔案儲存結構。
- 調整 TensorBoard 日誌的檔案儲存結構。
- 將 `extract_charset` 腳本更名為 `split_dataset`。
- 將 `RANDOM_SEED` 腳本參數更名為 `SPLIT_RANDOM_SEED`。
- 完善 README 文件內容。

### 📦 支援套件
- 更新 Python 支援至 3.13。
- 更新 clean-fid 支援至 0.1.25。
- 更新 tensorboard 支援至 2.20.0。
- 更新 torch 支援至 2.7.1。

---

## v1.0.0
- 2025 年 5 月 11 日

### ✨ 新增功能
- 釋出並開源字生字（HanziGen）v1.0.0 原始碼。