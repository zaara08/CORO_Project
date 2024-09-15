import datetime
import itertools
import os
import numpy as np
from pdb import set_trace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdb import set_trace

def plot_split_count(counts_train, counts_test, path, name='split_count', save_figure=True, overwrite=True):
    import matplotlib.pyplot as plt

    N = len(counts_train)
    train_means = [val for key, val in counts_train.items()]

    ind = np.arange(N) 
    width = 0.35       

    fig, ax = plt.subplots()
    rects1 = ax.barh(ind, train_means, width, color='r')

    test_means = [val for key, val in counts_test.items()]
    rects2 = ax.barh(ind + width, test_means, width, color='y')


    ax.set_xlabel('Counts')
    ax.set_title('Number of label occurences')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels([key for key, val in counts_train.items()], fontsize=8)

    ax.legend((rects1[0], rects2[0]), ('Train', 'Test'))

    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
            
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            fig.savefig(figure_path)
        else:
            print("Figure already existed under given name. Saved with current time stamp")
            figure_path = os.path.join(path, name + '_{date:%Y-%m-%d_%H-%M-%S}.jpg'.format(date=datetime.datetime.now()))
            fig.savefig(figure_path)
    plt.close()
    return

    def _autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    _autolabel(rects1)
    _autolabel(rects2)
    return

def plot_mean(mean, path, name='mean', ylabel='loss', save_figure=True, overwrite=True):
    
    fig = plt.figure()
    n = len(mean)
    epochs = np.arange(1, n+1, dtype=np.int32)
    plt.plot(epochs, mean)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)

    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
            
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            fig.savefig(figure_path)
        else:
            print("Figure already existed under given name. Saved with current time stamp")
            figure_path = os.path.join(path, name + '_{date:%Y-%m-%d_%H-%M-%S}.jpg'.format(date=datetime.datetime.now()))
            fig.savefig(figure_path)
    plt.close()
    return

def plot_multiple_mean(mean, path, labels, name='multiple_mean', ylabel='accuracy', save_figure=True, overwrite=True):
    plt.figure()
    M = mean.shape[0]
    N = mean.shape[1]
    epochs = np.arange(1, N+1)
    colors = plt.cm.hsv(np.linspace(0, 1, N)).tolist()
    
    for m in range(M):
        plt.plot(epochs, mean[m, :], color=colors[m], label=labels[m])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)

    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            plt.savefig(figure_path)
        else:
            plt.savefig(figure_path)

            print("Figure already existed under given name. Saved with ccurrent time stamp")
            figure_path = os.path.join(path, name + '{date:%Y-%m-%d_%H:%M:%S}.jpg'.format(date=datetime.datetime.now()))
            plt.savefig(figure_path)
    plt.close()
    return

def plot_confusion_matrix(cm, classes, path, name='confusion_matrix',
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, save_figure=True, overwrite=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:

        cm = np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])

    else:
        pass

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10, rotation=90)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            plt.savefig(figure_path)
        else:
            plt.savefig(figure_path)

            print("Figure already existed under given name. Saved with ccurrent time stamp")
            figure_path = os.path.join(path, name + '{date:%Y-%m-%d_%H:%M:%S}.jpg'.format(date=datetime.datetime.now()))
            plt.savefig(figure_path)
    plt.close()
    return

def plot_results(mean, std, path, name, save_figure=True, overwrite=True):

    fig = plt.figure()
    n = len(mean)
    epochs = np.arange(1, n+1, dtype=np.int32)
    plt.errorbar(epochs, mean, std)
    plt.xlabel('Epoch')
    plt.ylabel('Return')

    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
            
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            fig.savefig(figure_path)
        else:
            print("Figure already existed under given name. Saved with current time stamp")
            figure_path = os.path.join(path, name + '_{date:%Y-%m-%d_%H-%M-%S}.jpg'.format(date=datetime.datetime.now()))
        
            fig.savefig(figure_path)
    plt.close()
    return

def plot_multiple(mean, std, path, labels, name, save_figure=True, overwrite=True):

    plt.figure()
    M = mean.shape[0]
    N = mean.shape[1]
    epochs = np.arange(1, N+1)
    colors = plt.cm.hsv(np.linspace(0, 1, N)).tolist()
    
    for m in range(M):
        plt.errorbar(epochs, mean[m, :],  std[m, :], color=colors[m], label=labels[m])
        plt.xlabel('Epoch')
        plt.ylabel('Return')

    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            plt.savefig(figure_path)
        else:
            print("Figure already existed under given name. Saved with ccurrent time stamp")
            figure_path = os.path.join(path, name + '{date:%Y-%m-%d_%H:%M:%S}.jpg'.format(date=datetime.datetime.now()))
            plt.savefig(figure_path)
    plt.close()
    return

def save_statistics(mean, std, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    mean = np.asarray(mean)
    np.save(os.path.join(path, name + '_mean'), mean)

    std = np.asarray(std)
    np.save(os.path.join(path, name + '_std'), std)




def concat_frames(video_file, outdir):
  reader = imageio.get_reader(video_file)
  for i, img in enumerate(reader):
      if i == 0:
        concat_img = img
        continue
      else:
        concat_img = np.concatenate([concat_img, img], axis=1)
  plt.imsave(join(outdir, video_file.split('.mp4')[0] + '.jpg'), concat_img)


def concat_frames_nosave(frames):
  for i, img in enumerate(frames):
      if i == 0:
        concat_img = img
        continue
      else:
        concat_img = np.concatenate([concat_img, img], axis=2)
  return concat_img

def plot_image_from_se3_output(image_tensor):
    if image_tensor.device.type == 'cuda':
        image_tensor = image_tensor.cpu()
    matplotlib.use('TkAgg')
    image = image_tensor.squeeze(0).permute(1,2,0).detach().numpy()
    plt.imshow(image)
    plt.show()
    return

def plot_image_from_se3_input_output_pair(image_tensor, image_tensor_out):
    matplotlib.use('TkAgg')
    if image_tensor.device.type == 'cuda':
        image_tensor = image_tensor.cpu()
        image_tensor_out = image_tensor_out.cpu()

    image = image_tensor.squeeze(0).permute(1,2,0).detach().numpy()
    image_out = image_tensor_out.squeeze(0).permute(1,2,0).detach().numpy()
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image)
    ax[1].imshow(image_out)
    # Add title for each image
    ax[0].set_title('Input Image')
    ax[1].set_title('Output Image')

    plt.show()
    return

def plot_image_from_se3_input_output_gt(image_tensor, image_tensor_gt, image_tensor_out):
    matplotlib.use('TkAgg')
    if image_tensor.device.type == 'cuda':
        image_tensor = image_tensor.cpu()
        image_tensor_gt = image_tensor_gt.cpu()
        image_tensor_out = image_tensor_out.cpu()
    image = image_tensor.squeeze(0).permute(1,2,0).detach().numpy()
    image_out = image_tensor_out.squeeze(0).permute(1,2,0).detach().numpy()
    image_gt = image_tensor_gt.squeeze(0).permute(1,2,0).detach().numpy()
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(image)
    ax[1].imshow(image_gt)
    ax[2].imshow(image_out)
    # Add title for each image
    ax[0].set_title('Input Image')
    ax[1].set_title('Ground Truth Image')
    ax[2].set_title('Output Image')
    plt.show()
    return
