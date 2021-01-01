import numpy as np
import torch
import sys
import os
import pdb
#from sklearn.metrics import jaccard_similarity_score as jsc

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100,
                     fill='=', empty=' ', tip='>', begin='[', end=']', done="[DONE]", clear=True):
    """
    Print iterations progress.
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required : current iteration                          [int]
        total       - Required : total iterations                           [int]
        prefix      - Optional : prefix string                              [str]
        suffix      - Optional : suffix string                              [str]
        decimals    - Optional : positive number of decimals in percent     [int]
        length      - Optional : character length of bar                    [int]
        fill        - Optional : bar fill character                         [str] (ex: 'â– ', 'â–ˆ', '#', '=')
        empty       - Optional : not filled bar character                   [str] (ex: '-', ' ', 'â€¢')
        tip         - Optional : character at the end of the fill bar       [str] (ex: '>', '')
        begin       - Optional : starting bar character                     [str] (ex: '|', 'â–•', '[')
        end         - Optional : ending bar character                       [str] (ex: '|', 'â–', ']')
        done        - Optional : display message when 100% is reached       [str] (ex: "[DONE]")
        clear       - Optional : display completion message or leave as is  [str]
    """
    #pdb.set_trace()
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength
    if iteration != total:
         bar = bar + tip
    bar = bar + empty * (length - filledLength - len(tip))
    display = '\r{prefix}{begin}{bar}{end} {percent}%{suffix}' \
              .format(prefix=prefix, begin=begin, bar=bar, end=end, percent=percent, suffix=suffix)
    print(display, end=''),   # comma after print() required for python 2
    if iteration == total:      # print with newline on complete
        if clear:               # display given complete message with spaces to 'erase' previous progress bar
            finish = '\r{prefix}{done}'.format(prefix=prefix, done=done)
            if hasattr(str, 'decode'):   # handle python 2 non-unicode strings for proper length measure
                finish = finish.decode('utf-8')
                display = display.decode('utf-8')
            clear = ' ' * max(len(display) - len(finish), 0)
            print(finish + clear)
        else:
            print('')


def verbose(verboseLevel, requiredLevel, printFunc=print, *printArgs, **kwPrintArgs):
    """
    Calls `printFunc` passing it `printArgs` and `kwPrintArgs`
    only if `verboseLevel` meets the `requiredLevel` of verbosity.
    Following forms are supported:
        > verbose(1, 0, "message")
            >> message
        > verbose(1, 0, "message1", "message2")
            >> message1 message2
        > verbose(1, 2, "message")
            >>          <nothing since verbosity level not high enough>
        > verbose(1, 1, lambda x: print('MSG: ' + x), 'message')
            >> MSG: message
        > def myprint(x, y="msg_y", z=True): print('MSG_Y: ' + y) if z else print('MSG_X: ' + x)
        > verbose(1, 1, myprint, "msg_x", "msg_y")
            >> MSG_Y: msg_y
        > verbose(1, 1, myprint, "msg_x", "msg_Y!", z=True)
            >> MSG_Y: msg_Y!
        > verbose(1, 1, myprint, "msg_x", z=False)
            >> MSG_X: msg_x
        > verbose(1, 1, myprint, "msg_x", z=True)
            >> MSG_Y: msg_y
    """
    if verboseLevel >= requiredLevel:
        # handle cases when no additional arguments are provided (default print nothing)
        printArgs = printArgs if printArgs is not None else tuple([''])
        # handle cases when verbose is called directly with the object (ex: str) to print
        if not hasattr(printFunc, '__call__'):
            printArgs = tuple([printFunc]) + printArgs
            printFunc = print
        printFunc(*printArgs, **kwPrintArgs)


def print_flush(txt=''):
    print(txt)
    sys.stdout.flush()


if os.name == 'nt':
    import msvcrt
    import ctypes

    class _CursorInfo(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int),
                    ("visible", ctypes.c_byte)]


def hide_cursor():
    if os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = False
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


def show_cursor():
    if os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = True
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

def one_hot(labels,
            num_classes,
            device = None,
            dtype= None,
            eps = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) 

#CODE FOR ALL THE METRICS USED PRESENTED HERE
SMOOTH = 1e-6
def intersection_and_union(pred, label, num_class=7):
              
    iou_list = list()
    cwise_iou = list()
    pred = pred.view(-1)
    label = label.view(-1)
   
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(0,num_class):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = None
            cwise_iou.append(float("nan"))
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) /(float(union_now)+SMOOTH)
            iou_list.append(iou_now)
            cwise_iou.append(iou_now)

    return np.mean(iou_list),cwise_iou

def dice_coeff(true, input, eps=1e-7):
   
    C=7
    target_one_hot = one_hot(true, num_classes=C,
                                 device=input.device, dtype=input.dtype)
    input_soft = one_hot(input, num_classes=C,
                                 device=input.device, dtype=input.dtype)
    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims).float()
    cardinality = torch.sum(input_soft + target_one_hot, dims).float()

    dice_score = 2. * intersection / (cardinality + eps)
    return torch.mean(dice_score)

def accuracy_pixel(true, input, conf_mat, total , eps=1e-7):
    for cl in range(7): 
      
        conf_mat[cl,:] += ((true[:,cl,:,:].unsqueeze(1)*input)!=0).sum(dim=(2,3))[0,:].float()
        total[cl] += (true[:,cl,:,:]!=0).sum()
    return
