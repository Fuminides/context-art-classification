python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture vit --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./vit_base_type_best_model.pth.tar --patience 10 
python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture vit --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./vit_base_school_best_model.pth.tar --patience 10
python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture vit --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./vit_base_author_best_model.pth.tar --patience 10
python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture vit --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./vit_base_time_best_model.pth.tar --patience 10


python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture convnext --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./convnext_base_type_best_model.pth.tar --patience 10
python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture convnext --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./convnext_base_school_best_model.pth.tar --patience 10
python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture convnext --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./convnext_base_author_best_model.pth.tar --patience 10
python main.py --mode train --workers 0 --model kgm --att type --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture convnext --batch_size 32 --nepochs 100 --embedds bow --base base --resume ./convnext_base_time_best_model.pth.tar --patience 10


python main.py --mode train --workers 0 --model mtl --att all --dir_dataset ../../myvolumes/SemArt/SemArt/ --architecture convnext --batch_size 32 --nepochs 100 --embedds bow --base base --resume convnext_mtl_best_model.pth.tar

