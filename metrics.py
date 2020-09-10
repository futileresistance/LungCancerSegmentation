def conf_mat(y_true, y_pred, smooth=1e-3):
    y_pred[y_pred > 0.2] = 1
    y_pred[y_pred < 0.2] = 0
    tn, fp, fn, tp = confusion_matrix(np.ravel(y_true[:,:,:,0]), np.ravel(y_pred[:,:,:,0])).ravel()
    return tn, fp, fn, tp


def count_dice(y_true, y_pred, thresh=0.2):
    number_of_slices = y_true.shape[0]
    pred = y_pred
    mask = y_true
    results = []
    for i in range(number_of_slices):
        pred_i = pred[i,:,:,0]
        pred_i[pred_i > thresh] = 1
        pred_i[pred_i < thresh] = 0

        dice = dice_score(mask_i, pred_i)
        sess = tf.InteractiveSession()
        dice = dice.eval()
        sess.close()
        results.append(dice)
    return np.mean(results)