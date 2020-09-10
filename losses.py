import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))

def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice

def double_head_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    return mask_loss + contour_loss

def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_crossentropy(y_true, y_pred, weight0=1, weight1=2):
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1-keras.backend.epsilon())
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1-keras.backend.epsilon())
    logloss = -(y_true * keras.backend.log(y_pred) * weight1 + \
                (1 - y_true) * keras.backend.log(1 - y_pred) * weight0 )
    return keras.backend.mean(logloss, axis=-1)

def weighted_bce_loss(y_true, y_pred, weight=2.0):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def focal_loss(y_true, y_pred):
    gamma=0.75
    alpha=0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def double_loss(y_true, y_pred):
    focal = focal_loss(y_true, y_pred)
    dice = dice_coef_loss(y_true, y_pred)
    return focal + dice