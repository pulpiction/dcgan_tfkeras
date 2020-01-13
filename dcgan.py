import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import Dense, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape

import os
import argparse

from preprocess import load_batch
from imageio import imwrite

# Available GPUs and error logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print('GPU Available: ', gpu_available)

# Argparser for ease of use in terminal environments
parser = argparse.ArgumentParser(description='DCGAN')

parser.add_argument('--img-dir', type=str, default='./data/celebA', help='data directory training image live')
parser.add_argument('--out-dir', type=str, default='./output', help='data where output images are stored')
parser.add_argument('--mode', type=str, default='train', help='mode can be either train or test')
parser.add_argument('--restore-checkpoint', action='store_true', help='flag to resume training from a previously-saved checkpoint')
parser.add_argument('--z-dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--batch-size', type=int, default=128, help='image batch size for training and testing')
parser.add_argument('--num-data-threads', type=int, default=2, help='number of threads to use when loading and preprocessing training images')
parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs for training images for pass through')
parser.add_argument('--learn-rate', type=float, default=0.0002, help='learning rate for optimizer')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 parameter for Adam optimizer')
parser.add_argument('--num-gen-updates', type=float, default=2, help='number of generator updates per discriminator update to stabilize training')
parser.add_argument('--log-every', type=int, default=7, help='number of iterations for printing losses')
parser.add_argument('--save-every', type=int, default=500, help='number of iterations for saving network parameters')
parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0', help='device to run the computation given in tf convention')

args = parser.parse_args()

fid_module = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001])])

def fid_function(real_img, generated_img):
    img_size = (299, 299)
    real_resize = tf.image.resize(real_img, img_size)
    fake_resize = tf.image.resize(generated_img, img_size)
    fid_module.build([None, 299, 299, 3])
    
    real_features = fid_module(real_resize)
    fake_features = fid_module(fake_resize)
    return tfgan.eval.frechet_classifier_distance_from_activations(real_features, fake_features)


class generator(tf.keras.Model):
    def __init__(self):
        super(generator, self).__init__()

        # Random initializer
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        # Model, Optimizer, Loss Function
        self.model = tf.keras.Sequential([
            Dense(4*4*512, kernel_initializer=init, use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.0),
            Reshape([4,4,512]),
            Conv2DTranspose(256, (5,5), (2,2), kernel_initializer=init, 
                            padding='SAME', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.0),
            Conv2DTranspose(128, (5,5), (2,2), kernel_initializer=init,
                            padding='SAME', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.0),
            Conv2DTranspose(64, (5,5), (2,2), kernel_initializer=init,
                            padding='SAME', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.0),
            Conv2DTranspose(3, (5,5), (2,2), kernel_initializer=init, bias_initializer=init,
                            padding='SAME', activation='tanh')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
        self.loss = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def call(self, inputs):
        return self.model(inputs)

    @tf.function
    def loss_function(self, fake_output):
        loss =self.loss(tf.ones_like(fake_output), fake_output)
        return tf.reduce_mean(loss)

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()

        # Random initializer
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        # Model, Optimizer, Loss Function
        self.model = tf.keras.Sequential([
            Conv2D(64, (5,5), (2,2), padding='SAME', kernel_initializer=init),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (5,5), (2,2), kernel_initializer=init,
                   padding='SAME', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(256, (5,5), (2,2), kernel_initializer=init,
                   padding='SAME', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(512, (5,5), (2,2), kernel_initializer=init,
                   padding='SAME', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(1, kernel_initializer=init, bias_initializer=init, activation='sigmoid')
        ])  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
        self.loss = tf.keras.losses.BinaryCrossentropy()
    
    @tf.function
    def call(self, inputs):
        return self.model(inputs)

    @tf.function
    def loss_function(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

def train(generator, discriminator, dataset_iterator, train_manager):
    fid, batch_num = 0, 0
    for iteration, batch in enumerate(dataset_iterator):
        z = tf.random.normal([args.batch_size, args.z_dim])

        # Optimize generator
        for _ in range(args.num_gen_updates):
            with tf.GradientTape() as g_tape:
                gen_output = generator.call(z)
                fake_logits = discriminator.call(gen_output)
                g_loss = generator.loss_function(fake_logits)

            g_grad = g_tape.gradient(g_loss, generator.trainable_variables)
            generator.optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))

        # Optimize discriminator
        with tf.GradientTape() as d_tape:
            gen_output = generator.call(z)
            real_logits = discriminator.call(batch)
            fake_logits = discriminator.call(gen_output)
            d_loss = discriminator.loss_function(real_logits, fake_logits)

        d_grad = d_tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))

        if iteration % args.save_every == 0:
            train_manager.save()

            f = fid_function(batch, gen_output)
            print('FID (Inception Distance) %g' % f)

            fid = fid + f
            batch_num = batch_num + 1
    
    return fid / batch_num

def test(generator):
    z = tf.random.normal([args.batch_size, args.z_dim])
    img = generator.call(z).numpy()
    img = ((img / 2) - 0.5) * 255

    for i in range(args.batch_size):
        img_i = img[i]
        s = args.out_dir + '/' + str(i) + '.png'
        imgwrite(s, img_i)
    
    pass

def main():
    dataset_iterator = load_batch(args.img_dir, batch_size=args.batch_size, n_threads=args.num_data_threads)

    g = generator()
    d = discriminator()

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator=g, discriminator=d)
    train_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.restore_checkpoint or args.mode=='test':
        checkpoint.restore(train_manager.latest_checkpoint)
    
    try:
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                for epoch in range(args.num_epochs):
                    print('EPOCH %d : FID %f' % (epoch, train(g, d, dataset_iterator, train_manager)))
                    print('SAVING CHECKPOINT... ')
                    train_manager.save()
            if args.mode == 'test':
                test(g)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    main()
