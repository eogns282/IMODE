import os
import matplotlib.pyplot as plt


def vis_decay_traj(obs, pred, itv, epoch, exp_name, val=None):
    obs_x = obs[:, 0].cpu().detach().numpy()
    obs_y = obs[:, 1].cpu().detach().numpy()
    plt.scatter(obs_x[0], obs_y[0])  # starting points
    plt.plot(obs_x, obs_y)

    itv_x = itv[:, 0].cpu().detach().numpy()
    itv_y = itv[:, 1].cpu().detach().numpy()

    pred_x = pred[:, 0].cpu().detach().numpy()
    pred_y = pred[:, 1].cpu().detach().numpy()
    plt.plot(pred_x, pred_y, color='red')

    itv_mask = (itv_x != 0.0) | (itv_y != 0.0)
    plt.scatter(pred_x[itv_mask], pred_y[itv_mask], c='black')

    epoch = str(epoch)
    epoch = (3 - len(epoch)) * '0' + epoch

    path = './vis' + exp_name
    if not os.path.isdir(path):
        if not os.path.isdir('./vis'):
            os.mkdir('./vis')
        os.mkdir(path)

    if val is None:
        path += 'train/'
    else:
        path += 'val/'

    if not os.path.isdir(path):
        os.mkdir(path)

    plt.savefig(path + epoch + '.png', dpi=300)
    plt.close()


def vis_collision_traj(obs, pred, itv, epoch, exp_name, val=None):
    obs_x = obs[:, 0].cpu().detach().numpy()
    obs_y = obs[:, 1].cpu().detach().numpy()
    plt.scatter(obs_x[0], obs_y[0])  # starting points
    plt.plot(obs_x, obs_y)

    itv_x = itv[:, 0].cpu().detach().numpy()
    itv_y = itv[:, 1].cpu().detach().numpy()

    pred_x = pred[:, 0].cpu().detach().numpy()
    pred_y = pred[:, 1].cpu().detach().numpy()
    plt.plot(pred_x, pred_y, color='red')

    itv_mask = (itv_x != 0.0) | (itv_y != 0.0)
    plt.scatter(pred_x[itv_mask], pred_y[itv_mask], c='black')

    epoch = str(epoch)
    epoch = (3 - len(epoch)) * '0' + epoch

    path = './vis' + exp_name
    if not os.path.isdir(path):
        if not os.path.isdir('./vis'):
            os.mkdir('./vis')
        os.mkdir(path)

    if val is None:
        path += 'train/'
    else:
        path += 'val/'

    if not os.path.isdir(path):
        os.mkdir(path)

    plt.savefig(path + epoch + '.png', dpi=300)
    plt.close()