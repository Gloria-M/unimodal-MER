import numpy as np


def get_quadrant(measurements, value_range=(0, 1)):
    val_min, val_max = value_range
    val_mid = (val_max - val_min) / 2 + val_min

    valence, arousal = measurements
    if valence > val_mid:
        return 1 if arousal > val_mid else 4
    else:
        return 2 if arousal >= val_mid else 3


def get_quadrants_dict(valence_dict, arousal_dict):

    true_annotations = [measurement for measurement in zip(valence_dict['true_annotations'],
                                                           arousal_dict['true_annotations'])]
    pred_annotations = [measurement for measurement in zip(valence_dict['pred_annotations'],
                                                           arousal_dict['pred_annotations'])]

    true_quadrant = np.array([get_quadrant(measurement) for measurement in true_annotations])
    pred_quadrant = np.array([get_quadrant(measurement) for measurement in pred_annotations])

    quadrants_dict = {'true_annotations': true_quadrant,
                      'pred_annotations': pred_quadrant}

    quadrants_names = [1, 2, 3, 4]
    for quadrant in quadrants_names:

        q_pred = true_quadrant[np.where(pred_quadrant == quadrant)]
        q_true = true_quadrant[np.where(true_quadrant == quadrant)]
        q_perc = np.sum(q_pred == quadrant) / len(q_true) * 100
        quadrants_dict['{:d}'.format(quadrant)] = q_perc

    return quadrants_dict


def scale_measurement(measurement, current_range, new_range=(0, 1)):
    old_min, old_max = current_range
    new_min, new_max = new_range
    measurement_scaled = (measurement - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return measurement_scaled


def print_epoch(epoch, train_dict, validation_dict, dimension):

    if dimension == 'both':
        print_keys = {'valence_loss': 'Valence',
                      'arousal_loss': 'Arousal'}
    else:
        print_keys = {'loss': dimension.capitalize()}

    print('\nEPOCH {:d}'.format(epoch))
    print('-' * 35)

    print('   Train loss')
    for key, message in print_keys.items():
        print('  {:>12s} : {:.3f}'.format(message, train_dict[key][-1]))

    if validation_dict is not None:
        print('\n   Validation loss')
        for key, message in print_keys.items():
            print('  {:>12s} : {:.3f}'.format(message, validation_dict[key][-1]))


def print_test_results(valence_dict, arousal_dict, quadrants_dict):

    print('VALENCE')
    print('   MAE : {:.4f}'.format(valence_dict['mae']))
    print('   MSE : {:.4f}'.format(valence_dict['mse']))
    print()
    print('AROUSAL')
    print('   MAE : {:.4f}'.format(arousal_dict['mae']))
    print('   MSE : {:.4f}'.format(arousal_dict['mse']))
    print()
    print('Accuracy')
    for quadrant in range(1, 5):
        print('   QUADRANT {:d} : {:.2f}%%'.format(quadrant, quadrants_dict[quadrant]))


def print_params(args_dict):
    for key, val in args_dict.items():
        print(f'{key} : {val}')


def fail_format(fail_message):
    fail_flag = '===FAILED==='

    return "\n{:s}\n   {:s}\n{:s}\n".format(fail_flag, fail_message, fail_flag)


def success_format(success_message):
    success_flag = '===SUCCEEDED==='

    return "\n{:s}\n   {:s}\n{:s}\n".format(success_flag, success_message, success_flag)

