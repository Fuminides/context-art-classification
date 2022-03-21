python main.py --workers 0 --att author  --mode test --model mtl --embedds graph --model_path Models/mtl-author/graph_author_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy
python main.py --workers 0 --att type  --mode test --model mtl --embedds graph --model_path Models/mtl-type/graph_type_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy
python main.py --workers 0 --att time  --mode test --model mtl --embedds graph --model_path Models/mtl-time/graph_time_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy
python main.py --workers 0 --att school  --mode test --model mtl --embedds graph --model_path Models/mtl-school/graph_school_best_model.pth.tar --dir_dataset "../SemArt/" | grep Accuracy

