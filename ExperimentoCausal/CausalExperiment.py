import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import util
import pandas as pd
import numpy as np
from tensorflow.keras import layers
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math
from sklearn.decomposition import PCA
import os
from scipy import stats
from statsmodels.stats import weightstats as stests

tf.config.experimental_run_functions_eagerly(True)

ALL_DATA = "E:Corrected_FA/ALL_DATA/"
info_data = "idaSearch_8_01_2020.csv"
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
ROI_VS_AD = 0
ROI_VS_GEN = 1


def make_generator_model(latent_size, condition_size, output_size):
    """
    Función que devuelve un generador con el tamaño de entrada y salida indicados

    :param latent_size: Tamaño del vector de ruido aleatorio
    :param condition_size: Tamaño de la condición
    :param output_size: Tamaño de la variable a generar
    :return: Generador G(Condition, Z) --> Output
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(latent_size + condition_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(64, input_shape=(latent_size + condition_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(output_size, activation="linear"))

    return model


def generator_loss(fake_output):
    """
    Función para calcular el loss del generador

    :param fake_output: Output del discriminador para los ejemplos generados
    :return: Loss del generador
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_discriminator_model(input_size, condition_size):
    """
    Función que devuelve un discriminador con el tamaño de entrada indicados

    :param input_size: Tamaño de la valiable real o generada
    :param condition_size: Tamaño de la condición
    :return: Devuelve la predicción del discriminador, D(EjemploReal, Condición) o D(G(Condition, Z), Condición)
    """
    model = tf.keras.Sequential()

    model.add(layers.Dense(128, input_shape=(input_size + condition_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(64, input_shape=(input_size + condition_size,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    """
    Función para calcular el loss del discriminador

    :param real_output: Output del discriminador para los ejemplos reales
    :param fake_output: Output del discriminador para los ejemplos generados
    :return: Loss del discriminador
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def train_step(generator, discriminator, examples, labels, noise_dim, epoch):
    """
    Ciclo de entrenamiento de la GAN

    :param generator: Modelo generador
    :param discriminator: Modelo discriminador
    :param examples: Ejemplos reales
    :param labels: Etiquetas de los ejemplos reales
    :param noise_dim: Tamaño del vector de ruido aleatorio
    :param epoch: Número de el epoch actual
    :return: Loss del generador y discriminador
    """
    noise = tf.random.normal([labels.shape[0], noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Entrenamos el discriminador solo en los ciclos pares
        if epoch % 2 == 0:
            disc_train = False
        else:
            disc_train = True

        generated_examples = generator(
            tf.concat([
                noise,
                labels],
                axis=1),
            training=disc_train)

        real_output = discriminator(
            tf.concat([
                examples,
                labels],
                axis=1),
            training=disc_train)

        fake_output = discriminator(
            tf.concat([
                generated_examples,
                labels],
                axis=1),
            training=disc_train)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    if epoch % 2 == 0:
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss.numpy(), disc_loss.numpy()


def train(generator, discriminator, dataset, epochs, noise_dim):
    """
    Entrenamiento de la GAN

    :param generator: Modelo generador
    :param discriminator: Modelo discriminador
    :param dataset: Dataset de ejemplos reales
    :param epochs: Número de epoch
    :param noise_dim: Tamaño del vector de ruido aleatorio
    :return: Lista con el loss del generador y discriminador por cada epoch
    """
    history = []
    start = time.time()
    for epoch in range(epochs):

        gen_loss, disc_loss = 0, 0
        count = 0
        # TODO: pasar imagenes bien
        for examples_batch, labels in dataset:
            g, d = train_step(generator, discriminator, examples_batch, labels, noise_dim, epoch)
            gen_loss += g
            disc_loss += d
            count += 1

        if epoch % 10 == 0:

            history.append((disc_loss / count, gen_loss / count, epoch))

    history.append((disc_loss / count, gen_loss / count, epoch))

    return history


def generate_examples(generator, num_examples, noise_dim, condition):
    """
    Generar ejemplos mediante un generador

    :param generator: Modelo generador
    :param num_examples: Número de imágenes a generar
    :param noise_dim: Tamaño del vector de ruido aleatorio
    :param condition: Condición de los ejemplos a generar
    :return: Ejemplos generados
    """
    noise = tf.random.normal([num_examples, noise_dim], dtype="float32")
    labels = tf.convert_to_tensor(condition, dtype="float32")
    inp = tf.concat([noise, labels], axis=1)
    generated_image = generator(inp, training=False)
    return generated_image.numpy()


def correct_preds(true_labels, preds):
    """
    Función que evalúa las predicciones, devolviendo un vector de float en el que 1 representa una predicción
    correcta y 0 una incorrecta

    :param true_labels: Etiquetas reales
    :param preds: Predicciones
    :return: Vector de predicciones
    """
    correct_labels = []
    for i, true_label in enumerate(true_labels):
        correct_labels.append(int(preds[i]) == int(true_label))
    return np.array(correct_labels, dtype="float32")


def correct_probs(true_labels, probs):
    """
    Devuelve las probabilidades que el modelo ha dado a la clase correcta

    :param true_labels: Etiquetas reales
    :param probs: Probabilidad de cada clase a cada ejemplo
    :return: Probabilidades de la clase correcta de cada ejemplo
    """
    correct_probs = []
    for i, true_label in enumerate(true_labels):
        correct_probs.append(probs[i, int(true_labels[i])])
    return np.array(correct_probs, dtype="float32")


def C2ST(condition, real, generator, noise_dim, train_test_prop, test_type):
    """
    Realiza un C2ST dado un generador e imágenes reales, evalúa el generador intentando distinguir entre los ejemplos
    generados y los reales.

    :param condition: Condición de los ejemplos (Etiquetas)
    :param real: Ejemplos reales
    :param generator: Modelo generador
    :param noise_dim: Tamaño del vector de ruido aleatorio
    :param train_test_prop: Proporción entre el conjunto de entrenamiento y test
    :param test_type: Tipo de test, si el test utiliza las condiciones o no
    :return: Puntuación recibida por el generador
    """

    n_images = real.shape[0]

    fake = generate_examples(generator, n_images, noise_dim, condition)
    fake_labels = np.zeros((n_images, 1), dtype="float32")
    real_labels = np.ones((n_images, 1), dtype="float32")
    labels = np.concatenate((real_labels, fake_labels), axis=0)

    dataset = np.concatenate((real, fake), axis=0)
    if test_type == ROI_VS_AD:
        conditions = np.concatenate((condition, condition), axis=0)
        dataset = np.concatenate((dataset, conditions), axis=1)

    n = dataset.shape[0]

    random_range = np.random.permutation(n)

    train_indexes = random_range[:int(train_test_prop * n)]
    n_train = train_indexes.shape[0]
    test_indexes = random_range[int(train_test_prop * n):]
    n_test = test_indexes.shape[0]

    dataset_train = dataset[train_indexes, :]
    labels_train = labels[train_indexes, :].reshape(-1)

    dataset_test = dataset[test_indexes, :]
    labels_test = labels[test_indexes, :].reshape(-1)

    KNN_classifier = KNeighborsClassifier(n_neighbors=int(math.sqrt(n_train)))

    KNN_classifier.fit(dataset_train, labels_train)
    predictions = KNN_classifier.predict(dataset_test)
    return accuracy_score(labels_test, predictions), \
           correct_probs(labels_test, KNN_classifier.predict_proba(dataset_test))


def train_gan(condition, examples, noise_dim, n_epoch, batch_size):
    """
    Entrenamiento de la GAN

    :param condition: Condición para la generación
    :param examples: Ejemplos reales
    :param noise_dim: Tamaño del vector de ruido
    :param n_epoch: Número de epoch
    :param batch_size: Tamaño del batch
    :return: Generador entrenado e historia de la GAN
    """
    dataset = tf.data.Dataset.from_tensor_slices((examples.astype("float32"), condition.astype("float32"))).shuffle(
        condition.shape[0]).batch(batch_size)
    generator = make_generator_model(YtoX_noise_dim, condition.shape[1], examples.shape[1])
    discriminator = make_discriminator_model(examples.shape[1], condition.shape[1])

    history = train(generator, discriminator, dataset, n_epoch, noise_dim)

    return generator, history


def normalize(dataset):
    """
    Normalizar dataset

    :param dataset: Dataset a normalizar
    :return: Dataset Normalizado
    """
    dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
    return dataset


def load_gene(gene_path):
    """
    Cargar Gen

    :param gene_path: Dirección de los datos del gen
    :return: Dataset con los datos genéticos reducidos con PCA y tamaño de los genes reducidos
    """
    df_gen = pd.read_csv(gene_path, sep=" ").drop(["FID", "PHENOTYPE"], axis=1).fillna(-1)
    gene_values = df_gen.values[:, 1:]
    subjects = df_gen.values[:, 0].reshape(-1, 1)
    pca = PCA(n_components=min(50, gene_values.shape[1]), svd_solver="full")
    pca.fit(gene_values.T)
    pca_components = normalize(pca.components_.T)
    genetic_data = pd.DataFrame(np.concatenate((subjects, pca_components), axis=1))
    pca_components_size = pca_components.shape[1]

    return genetic_data, pca_components_size


def load_ROI(roi_path):
    """
    Cargar región de interés

    :param roi_path: Dirección de los datos de la ROI
    :return: Dataset de la ROI reducida mediante FPCA
    """
    AD_CN, groups = util.obtain_data_files(ALL_DATA, info_data)

    imgs = []
    labels = []
    for key in AD_CN.keys():
        for img in AD_CN[key]:
            imgs.append(img)
            labels.append(key)

    image_labels = pd.DataFrame({"Image": imgs, "Label": labels})

    fpcas_temporal_right = pd.read_csv(roi_path, index_col=0)
    images_size = fpcas_temporal_right.shape[1] - 1

    df_temporal_right = fpcas_temporal_right.merge(image_labels, left_on="file", right_on="Image")
    data = df_temporal_right.copy()
    data["SubjectID"] = data["file"].apply(lambda x: x[5:15])
    data = data.drop(["file", "Image"], axis=1)
    return data, images_size


def ROI_vs_gene_dataset(roi_path, gene_path):
    """
    Creación del dataset para el experimento ROI vs Genes
    :param roi_path: Dirección del ROI
    :param gene_path: Dirección del Gen
    :return: dataset de roi y dataset de genes
    """
    # Obtenemos los diccionarios con los nombres de los ficheros que contienen las imágenes
    data, images_size = load_ROI(roi_path)

    # Cargando datos geneticos
    genetic_data, gene_size = load_gene(gene_path)

    # Unimos los datos
    data = genetic_data.merge(data, how="inner", left_on=genetic_data[0], right_on=data["SubjectID"])
    data = data.drop(["Label", 0, "key_0", "SubjectID"], axis=1).astype("float32")

    data = data.sample(frac=1)  # Permutamos el dataset

    roi_data = normalize(data.iloc[:, gene_size:].values)
    gene_data = data.iloc[:, :gene_size].values

    return roi_data, gene_data


def ROI_vs_AD_dataset(roi_path):
    """
    Creación del dataset de ROI vs AD

    :param roi_path: Dirección del ROI
    :return: dataset de roi
    """
    # Obtenemos los diccionarios con los nombres de los ficheros que contienen las imágenes
    data, images_size = load_ROI(roi_path)

    # Unimos los datos
    data = data.sample(frac=1)  # Permutamos el dataset
    roi_data = normalize(data.drop(["Label", "SubjectID"], axis=1).values.astype("float32"))
    label = data["Label"].apply(lambda x: 1 if x == "AD" else 0).values.reshape((-1, 1)).astype("float32")

    return roi_data, label


def causal_experiment(X, Y, test_type):
    """
    Experimento causal

    :param X: Variable X
    :param Y: Variable Y
    :param test_type: Tipo de test, si es ROI_VS_AD se utilizan las condiciones en el test
    :return: Puntuaciones del test y sigma para calcular el p-valor
    """
    XtoY_generator, XtoY_history = train_gan(X, Y, XtoY_noise_dim, N_EPOCH, BATCH_SIZE)

    YtoX_generator, YtoX_history = train_gan(Y, X, YtoX_noise_dim, N_EPOCH, BATCH_SIZE)

    acc_x_y, correct_preds_x_y = C2ST(X, Y, XtoY_generator, XtoY_noise_dim, TRAIN_TEST_PROP, test_type)

    acc_y_x, correct_preds_y_x = C2ST(Y, X, YtoX_generator, YtoX_noise_dim, TRAIN_TEST_PROP, test_type)

    n_test = correct_preds_x_y.shape[0]

    sigma2 = (0.5 / n_test) - 2 * np.cov(correct_preds_x_y, correct_preds_y_x)[0, 1]

    return acc_x_y, acc_y_x, np.abs(acc_x_y - acc_y_x), np.abs(sigma2)


TRAIN_TEST_PROP = 0.7
BATCH_SIZE = 64
N_EPOCH = 1000
YtoX_noise_dim = 50
XtoY_noise_dim = 50

ROIs = [("Ventricle", "FPCAs/FPCAS_Ventricle.csv"), ("Temporal_Left", "FPCAs/FPCAS_Left_Temporal_Lobe.csv"),
        ("Temporal_Right", "FPCAs/FPCAS_Right_Temporal_Lobe.csv")]
ROIs = [("Caudate", "FPCAs/FPCAS_Caudate.csv"),
        ("Cerebellum ", "FPCAs/FPCAs_Cerebellum.csv"),
        ("FPCAs_Frontal_Lobe ", "FPCAs/FPCAs_Frontal_Lobe.csv"),
        ("FPCAs_Insula", "FPCAs/FPCAs_Insula.csv"),
        ("FPCAs_Occipital_Lobe", "FPCAs/FPCAs_Occipital_Lobe.csv"),
        ("FPCAs_Parietal_Lobe", "FPCAs/FPCAs_Parietal_Lobe.csv"),
        ("FPCAs_Putamen", "FPCAs/FPCAs_Putamen.csv"),
        ("FPCAs_Temporal_Lobe", "FPCAs/FPCAs_Temporal_Lobe.csv"),
        ("FPCAs_Thalamus", "FPCAs/FPCAs_Thalamus.csv"),
        ("Ventricle", "FPCAs/FPCAS_Ventricle.csv"),
        ("Temporal_Left", "FPCAs/FPCAS_Left_Temporal_Lobe.csv"),
        ("Temporal_Right", "FPCAs/FPCAS_Right_Temporal_Lobe.csv")]

GENES_DIRECTORY = "GenesRAW"
GENES = ([(f[:len(f) - 4], os.path.join(GENES_DIRECTORY, f)) for f in os.listdir(GENES_DIRECTORY)])
# X, Y = ROI_vs_gene_dataset("FPCAS_Ventricle.csv", "GenesRAW/FOLR2.raw")
start = time.time()
roi_output_file = "causalidad_roi_nuevas.csv"
gene_output_file = "causalidad_genes_probs.csv"

# Test de ROI vs AD
with open(roi_output_file, "w") as f:
    f.write("ROI,X_Y,Y_X,sigma2\n")

for roi in ROIs:
    start = time.time()
    roi_name = roi[0]
    roi_path = roi[1]
    print("empieza", roi_name)
    X, Y = ROI_vs_AD_dataset(roi_path)
    acc_x_y, acc_y_x, T, sigma2 = causal_experiment(X, Y, ROI_VS_AD)
    print("X_Y:", acc_x_y, "Y_X:", acc_y_x)
    print("Tiempo total : ", time.time() - start)
    with open(roi_output_file, "a") as f:
        f.write(f"{roi_name},{acc_x_y},{acc_y_x},{sigma2}\n")

# Test de ROI vs genes
with open(gene_output_file, "w") as f:
    f.write("ROI,GEN,X_Y,Y_X,sigma2\n")

for roi in ROIs:
    roi_name = roi[0]
    roi_path = roi[1]
    for gene in GENES:
        start = time.time()
        gene_name = gene[0]
        gene_path = gene[1]
        print("empieza", roi_name, gene_name)
        X, Y = ROI_vs_gene_dataset(roi_path, gene_path)
        acc_x_y, acc_y_x, T, sigma2 = causal_experiment(X, Y)
        print("X_Y:", acc_x_y, "Y_X:", acc_y_x)
        print("Tiempo total : ", time.time() - start)
        with open(gene_output_file, "a") as f:
            f.write(f"{roi_name},{gene_name},{acc_x_y},{acc_y_x},{sigma2}\n")
