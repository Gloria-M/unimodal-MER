import os

from config.config import ArgsFromJSON

from data_preprocessing import DataPreprocessor
from visualize import Visualizer
from train import Trainer
from test import Tester
from utility_functions import *


def preprocess_data(args):

    if not os.path.exists(args.font_dir):
        os.mkdir(args.font_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    data_preprocessor = DataPreprocessor(args)
    visualizer = Visualizer(args.font_dir, args.plots_dir)

    data_preprocessor.get_data_info()
    data_preprocessor.get_waveforms()
    data_preprocessor.augment_quadrants()
    data_preprocessor.make_train_test_sets()

    visualizer.visualize_data_distribution(data_preprocessor.annotations, data_preprocessor.quadrants)
    visualizer.visualize_dimensions_distribution(data_preprocessor.annotations)


def run_train(args):

    if not os.path.exists(args.font_dir):
        os.mkdir(args.font_dir)
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    visualizer = Visualizer(args.font_dir, args.plots_dir)
    trainer = Trainer(args)

    for epoch in range(trainer.num_epochs):
        if trainer.dimension == 'both':
            trainer.train_2d()
            trainer.validate_2d()
        else:
            trainer.train_1d()
            trainer.validate_1d()

        if (epoch + 1) % trainer.log_interval == 0 or (epoch + 1) == trainer.num_epochs:
            print_epoch(epoch + 1, trainer.train_dict, trainer.test_dict, trainer.dimension)

        if (epoch + 1) % args.decay_interval == 0:
            trainer.update_learning_rate()

    visualizer.plot_losses(trainer.train_dict, trainer.test_dict, trainer.dimension)
    trainer.save_model()


def run_test(args):

    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)

    visualizer = Visualizer(args.font_dir, args.plots_dir)
    tester = Tester(args)

    if tester.dimension == 'both':
        tester.load_model_2d()
        tester.test_2d()
    else:
        tester.load_model_1d()
        tester.test_1d()

    if tester.dimension == 'both':
        title = '2D Model'
    else:
        title = '1D Models'

    visualizer.plot_quadrant_predictions(tester.valence_dict, tester.arousal_dict, tester.quadrants_dict, title)
    visualizer.plot_valence_predictions(tester.valence_dict, title)
    visualizer.plot_arousal_predictions(tester.arousal_dict, title)


if __name__ == '__main__':

    args_from_json = ArgsFromJSON('config/config_file.json')
    args_from_json.get_args_from_dict()
    args = args_from_json.parser.parse_args()

    print('\n\n')
    print_params(vars(args))
    print('\n\n')

    if args.mode == 'preprocess':
        preprocess_data(args)

    elif args.mode == 'train':
        run_train(args)

    elif args.mode == 'test':
        run_test(args)
