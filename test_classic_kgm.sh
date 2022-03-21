python main.py --att author  --mode test --model kgm --embedds graph --model_path Models/kgm-author/graph_author_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy
python main.py --att type  --mode test --model kgm --embedds graph --model_path Models/kgm-type/graph_type_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy
python main.py --att time  --mode test --model kgm --embedds graph --model_path Models/kgm-time/graph_time_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy
python main.py --att school  --mode test --model kgm --embedds graph --model_path Models/kgm-school/graph_school_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy

