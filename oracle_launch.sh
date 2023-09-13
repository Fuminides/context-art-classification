python main.py --mode train --workers 0 --model kgm --att all --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture vit --batch_size 64 --nepochs 300 --embedds bow --resume ./Models/vit_fcm_best_model.pth.tar >salidas/train_vit.txt
python main.py --mode test --workers 0 --model kgm --att all --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture convnext --batch_size 64 --nepochs 300 --embedds bow --model_path ./Models/convnext_fcm_best_model.pth.tar >salidas/test_vit.txt

python main.py --mode train --workers 0 --model kgm --att all --dir_dataset ./../myvolumes/SemArt/SemArt/ --architecture convnext --batch_size 64 --nepochs 300 --embedds bow --resume ./Models/convnext_fcm_best_model.pth.tar >salidas/train_convnext.txt
python main.py --mode test --workers 0 --model kgm --att all --dir_dataset ./../myvolumes/SemArt/SemArt/ --architecture vit --batch_size 64 --nepochs 300 --embedds bow --resume ./Models/vit_fcm_best_model.pth.tar >salidas/test_convnext.txt

