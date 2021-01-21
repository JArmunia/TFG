import math
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import util

import multiprocessing as mp


def sliding_window_batch(model, images, mean, std, muestra):
    k_size = 3

    # Obtenemos P0
    images_tensor = tf.constant(images.reshape(images.shape[0], 91, 109, 91, 1))
    p0 = model.predict(images_tensor)

    # Creamos la matriz de relevancia
    relevance_matrix = np.zeros(
        (images.shape[0], images.shape[1] - 2, images.shape[2] - 2, images.shape[3] - 2, images.shape[4]), "float32")

    # Para medir el progreso
    total = np.count_nonzero(muestra)
    count = 0

    new_images = images.copy()
    beforee = time.time()
    tiempo_predict = 0.
    tiempo_resto = 0.
    for i in range(images.shape[1] - (k_size - 1)):
        for j in range(images.shape[2] - (k_size - 1)):
            for k in range(images.shape[3] - (k_size - 1)):
                if muestra[i + 1, j + 1, k + 1] != 0:
                    b = time.time()
                    # Creamos el parche
                    kernel = np.random.normal(mean, std, [images.shape[0], 3, 3, 3, 1])

                    # Recuperamos las imágenes originales restaurando el parche anterior con los valores originales,
                    # así no hay que copiarlas todas
                    new_images[:, i - 1: (i - 1 + 4), j - 1: (j - 1 + 4), k - 1: (k - 1 + 4)] = images[:,
                                                                                                i - 1: (i - 1 + 4),
                                                                                                j - 1: (j - 1 + 4),
                                                                                                k - 1: (k - 1 + 4)]
                    # Aplicamos el parche nuevo
                    new_images[:, i: (i + 3), j: (j + 3), k: (k + 3)] = kernel

                    # Obtenemos P1
                    b_predict = time.time()
                    p1 = model.predict(tf.constant(new_images.reshape(images.shape[0], 91, 109, 91, 1)))
                    a_predict = time.time()
                    # Aplicamos la fórmula y ponemos el resultado en la matriz de relevancia
                    relevance_matrix[:, i, j, k] = np.log((p0 / (1 - p0)) / (p1 / (1 - p1)))

                    count += 1
                    a = time.time()
                    t_predict = a_predict - b_predict
                    t_demas = (a - b) - t_predict
                    tiempo_predict += t_predict
                    tiempo_resto += t_demas
                    if count % 100 == 0:
                        print(str(count * 100 / total) + " % ", time.time() - beforee, " s", " Tiempo predict: ",
                              tiempo_predict, " Tiempo resto: ", tiempo_resto)
                        tiempo_resto = 0.
                        tiempo_predict = 0.

    return relevance_matrix


def sliding_window_batch_stride(model, images, mean, std, muestra, stride):
    k_size = 3

    # Obtenemos P0
    images_tensor = tf.constant(images.reshape(images.shape[0], 91, 109, 91, 1))
    p0 = model.predict(images_tensor)

    # Creamos la matriz de relevancia
    relevance_matrix = np.zeros(
        (images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4]), "float32")

    # Para medir el progreso
    total = np.count_nonzero(muestra) / (stride ** 3)
    count = 0

    new_images = images.copy()
    beforee = time.time()
    tiempo_predict = 0.
    tiempo_resto = 0.
    range_i = int(images.shape[1] / stride)
    range_j = int(images.shape[2] / stride)
    range_k = int(images.shape[3] / stride)
    print(range_i, range_j, range_k)
    print("R i: ", range_i, "R j: ", range_j, "R k: ", range_k, "Total: ", total, range_i * range_j * range_k)
    for i in range(1, range_i):
        for j in range(1, range_j):
            for k in range(1, range_k):
                # Center of patch
                patch_i = stride * i
                patch_j = stride * j
                patch_k = stride * k

                if muestra[patch_i, patch_j, patch_k] != 0:
                    b = time.time()
                    # Creamos el parche
                    kernel = np.random.normal(mean, std, [images.shape[0], k_size, k_size, k_size, 1])

                    new_images = images.copy()

                    # Aplicamos el parche nuevo
                    new_images[:, patch_i - 1: (patch_i + 2),
                    patch_j - 1: (patch_j + 2),
                    patch_k - 1: (patch_k + 2)] = kernel

                    # Obtenemos P1
                    b_predict = time.time()
                    p1 = model.predict(tf.constant(new_images.reshape(images.shape[0],
                                                                      images.shape[1],
                                                                      images.shape[2],
                                                                      images.shape[3], 1)))
                    a_predict = time.time()
                    d = np.log((p0 / (1 - p0)) / (p1 / (1 - p1)))
                    # d[np.isnan(d)] = 0

                    # print("______________", p0.shape[0])
                    # for _ in range(p0.shape[0]):
                    #     print("P0: ", p0[_], " P1: ", p1[_], " d: ", d[_], type(d), _)
                    # Aplicamos la fórmula y ponemos el resultado en la matriz de relevancia
                    value_patch = np.ones((images.shape[0], stride, stride, stride, 1), dtype='float32') * \
                                  d.reshape(images.shape[0], 1, 1, 1, 1)
                    try:
                        relevance_matrix[:, patch_i - 1: (patch_i + 2),
                        patch_j - 1: (patch_j + 2),
                        patch_k - 1: (patch_k + 2), :] = value_patch

                    except ValueError as e:
                        print(relevance_matrix.shape, value_patch.shape)
                        print(e)
                        print("Error")
                        print("i", i, range_i, "j", j, range_j, "k", k, range_k)
                        print("Current: ", patch_i, patch_j, patch_k)

                    count += 1
                    a = time.time()
                    t_predict = a_predict - b_predict
                    t_demas = (a - b) - t_predict
                    tiempo_predict += t_predict
                    tiempo_resto += t_demas
                    if count % 100 == 0:
                        print(str(count * 100 / total) + " % ", time.time() - beforee, " s", " Tiempo predict: ",
                              tiempo_predict, " Tiempo resto: ", tiempo_resto)
                        tiempo_resto = 0.
                        tiempo_predict = 0.
                        np.save("Relevance_in_progress", relevance_matrix)

    return relevance_matrix


if __name__ == '__main__':

    ALL_DATA = "E:Corrected_FA/ALL_DATA/"
    info_data = "idaSearch_8_01_2020.csv"
    model_path = "model_loss_0.3_9999_0.1_0.003_2.h5"

    # Obtenemos los diccionarios con los nombres de los ficheros que contienen las imágenes
    AD_CN, groups = util.obtain_data_files(ALL_DATA, info_data)

    # Cargamos las imágenes
    CN_imgs = np.array(util.load_data(ALL_DATA, AD_CN["CN"]), dtype='float32')

    AD_imgs = util.load_data(ALL_DATA, AD_CN["AD"])

    # Extendemos la clase con menos ejemplos
    AD_imgs = np.array(util.extend_class(AD_imgs, len(CN_imgs)), dtype='float32')

    test_percentaje = 0.1
    val_percentaje = 0.1

    test_idx = math.floor(test_percentaje * len(CN_imgs))
    val_idx = math.floor(val_percentaje * len(CN_imgs)) + test_idx

    train_imgs = np.concatenate([CN_imgs[val_idx:], AD_imgs[val_idx:]])
    train_imgs = train_imgs.reshape((train_imgs.shape[0], 91, 109, 91, 1))

    # Obtenemos media y desviación del conjunto de entrenamiento con el que se ha entrenado el modelo
    mean = train_imgs.mean()
    std = train_imgs.std()

    # Grupos de puntos temporales
    etiquetas_grupos = ['ADNI2 Month 6-New Pt', 'ADNI2 Year 1 Visit',
                        'ADNI2 Year 2 Visit', 'ADNI2 Screening MRI-New Pt']

    data_grupos = {}

    # Obtenemos las imágenes de todos los grupos
    for etiqueta in etiquetas_grupos:
        arr = np.array(util.load_data(ALL_DATA, groups["AD " + etiqueta] + groups["CN " + etiqueta]), dtype='float32')
        arr = arr.reshape((arr.shape[0], 91, 109, 91, 1))
        # arr = (arr - mean) / std  # Normalizamos
        print(etiqueta, "Shape: ", arr.shape, "AD: ", len(groups["AD " + etiqueta]), " CN: ",
              len(groups["CN " + etiqueta]))
        data_grupos[etiqueta] = arr

    model = keras.models.load_model(model_path)  # roll back to best model
    model.compile(optimizer=keras.optimizers.Adam(3e-6), loss=tf.keras.losses.BinaryCrossentropy())

    for etiqueta in etiquetas_grupos:
        images = data_grupos[etiqueta]
        images_tensor = tf.constant(images.reshape(images.shape[0], 91, 109, 91, 1))
        # images_labels = np.zeros((96, 1))
        # images_labels[:46] = 1
        # print(model.evaluate(x=images_tensor, y=images_labels))
        preds = model.predict(images_tensor)
        preds = preds > 0.5
        print(preds.sum())

        rel_matrix = sliding_window_batch_stride(model, images, mean, std, CN_imgs[0], 3)

        np.save(f"Relevance_matrix_{etiqueta}", rel_matrix)
