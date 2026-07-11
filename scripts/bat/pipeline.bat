@echo off

call scripts\bat\analyze_font.bat
call scripts\bat\prepare_dataset.bat
call scripts\bat\split_dataset.bat
call scripts\bat\train_vqvae.bat
call scripts\bat\train_ldm.bat
call scripts\bat\compute_metrics.bat
call scripts\bat\inference.bat
call scripts\bat\convert_to_svg.bat