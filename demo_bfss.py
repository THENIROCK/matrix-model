import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_probability as tfp
from obs import *
from train import *
from wavefunc import *
from algebra import *
from dist import *
from bent_identity import *


"""
This is the script for BFSS variational energies.
This is adapted from the mini-BMN demo to use the BFSS Hamiltonian instead.
The BFSS Hamiltonian is: H = (1/2) tr(ΠᵢΠᵢ) - (1/4) tr([Xᵢ, Xⱼ]²)
"""

def truncate(data, lower=0.2, upper=0.8):
    # sorts and removes the exceptional (large or small) values in the list to reduce variance
    s, l = sorted(data), len(data)
    return s[int(lower * l) : int(upper * l)]

def flow_type(string):
    if string.lower() != "nf" and string.lower() != "maf":
        raise argparse.ArgumentTypeError("supported architectures: nf or maf")
    return string.lower()

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('flow', type=flow_type)
parser.add_argument('N', type=int, help='size of the matrices')
parser.add_argument('--prof', action='store_true', help='the flag used to profile the code')
parser.add_argument('-b', type=int, dest='batch_size', default=1000, help='batch size')
parser.add_argument('-s', type=int, dest='num_steps', default=10000, help='number of steps in training')
parser.add_argument('-f', type=int, dest='num_fermions', default=0, help='number of fermions')
parser.add_argument('-r', type=int, dest='rank', default=1, help='rank of the free fermion decomposition')
parser.add_argument('-i', dest='input_path', default=None, help='path of the model to restore')
parser.add_argument('-l', type=int, dest='num_layers', default=1, help='number of hidden layers of neural networks')
parser.add_argument('-n', type=int, dest='num_mix', default=1, help='number of distributions in the mixture')
# get the arguments as variables
args = parser.parse_args()
print(args)
globals().update(vars(args))
# spec
scope = flow + "N" + str(N) + "f" + str(num_fermions) + "r" + str(rank) + "l" + str(num_layers) + "n" + str(num_mix)
name = "bfss_" + flow + "N" + str(N) + "f" + str(num_fermions) + "r" + str(rank) + "l" + str(num_layers) + "n" + str(num_mix)
algebra = SU(N)
# build the model
with tf.variable_scope(scope):
    bosonic_dim = 2 * algebra.dim
    fermionic_dim = 2 * algebra.dim
    if flow == "nf":
        loc = tf.get_variable(name="loc", shape=[bosonic_dim, num_mix, 1], dtype=tf.float32)
        log_scale = tf.get_variable(name="log_scale", shape=[bosonic_dim, num_mix, 1], dtype=tf.float32)
        logits = tf.get_variable(name="logits", shape=[bosonic_dim, num_mix], dtype=tf.float32)
        vectorizer = Vectorizer(algebra, tfp.bijectors.Softplus())
        bosonic_wavefunc = NormalizingFlow([Mixture(
                                                [Affine(Normal(), loc=loc[i][j], log_scale=log_scale[i][j]) for j in range(num_mix)], 
                                                logits=logits[i]) 
                                            for i in range(bosonic_dim)], num_layers, BentIdentity(), None)
    else:
        num_pos = (algebra.N - 1) * 2
        dist = [Mixture([Affine(Gamma(), loc=tf.constant([0.0]))] * num_mix)] * num_pos
        dist = dist + [Mixture([Affine(Normal())] * num_mix)] * (bosonic_dim - num_pos)
        vectorizer = Vectorizer(algebra, tfp.bijectors.Identity())
        bosonic_wavefunc = Autoregressive(dist, bosonic_dim, num_layers, None)
    fermionic_wavefunc = FermionicWavefunction(algebra, bosonic_dim, 2, num_fermions, rank, 10, num_layers)
    wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
    bosonic = wavefunc.sample(batch_size)
    log_norm, _ = wavefunc(bosonic)
# observables
radius = tf.sqrt(matrix_quadratic_potential(bosonic) / N)
gauge = matrix_SUN_adjoint_casimir(wavefunc, bosonic)
rotation = miniBMN_SO3_casimir(wavefunc, bosonic)
energy = BFSS_bosonic_energy(wavefunc, bosonic)
# training
print("Training BFSS model...")
output_path = "results/" + name + "/"
obs = minimize(scope, log_norm, energy, num_steps,
                {"r": radius, "gauge": gauge, "rotation": rotation}, 
                lr=1e-3, restore_path=input_path, output_path=output_path, profiling=prof, save_every=num_steps)
obs = minimize(scope, log_norm, energy, num_steps,
                {"r": radius, "gauge": gauge, "rotation": rotation}, 
                lr=2e-4, restore_path=output_path, output_path=output_path, save_every=num_steps)
obs = minimize(scope, log_norm, energy, num_steps,
                {"r": radius, "gauge": gauge, "rotation": rotation}, 
                lr=4e-5, restore_path=output_path, output_path=output_path, save_every=num_steps)
# load the model
model_path = output_path + "model.ckpt"
variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
saver = tf.train.Saver(variables)
sess = tf.compat.v1.Session()
# Try to restore, but don't fail if model doesn't exist (it might not have been saved yet)
if not restore(sess, saver, model_path, True):
    print("Warning: Could not load model. This might be because training was too short.")
    print("For evaluation, you need to run training with more steps (e.g., -s 1000 or more).")
    exit(0)
# evaluation
print("Evaluating BFSS model...")
mean = []
std = []
mean_r = []
std_r = []
progbar = tf.keras.utils.Progbar(num_steps)
for i in range(num_steps):
    res, rad = sess.run([energy, radius])
    mean.append(np.mean(res))
    std.append(np.std(res))
    mean_r.append(np.mean(rad))
    std_r.append(np.std(rad))
    progbar.update(i + 1)
mean, std = truncate(mean), truncate(std)
print("Mean energy:", np.mean(mean))
print("MC uncertainty:", np.mean(std) / np.sqrt(num_steps * batch_size))
print("Mean radius:", np.mean(mean_r))
print("Stddev radius:", np.mean(std_r)) 