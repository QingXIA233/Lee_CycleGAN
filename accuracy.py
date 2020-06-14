'''Contains the accuracies of the discriminators for real and fake data'''
import tensorflow as tf

def Acc_real(real_image_logits):
	'''Compute the accuracy of the discriminators for real data.
	Args:
		real_image_logits: the output of the discriminator for
			real images.

	Returns:
		The accuracy of the discriminator for real images.'''

	pro_real_logits = tf.equal(tf.round(real_image_logits), tf.ones_like(real_image_logits))

	acc_real = tf.reduce_mean(tf.cast(pro_real_logits, tf.float32))

	return acc_real

def Acc_fake(fake_image_logits):
	'''Compute the accuracy of the discriminators for fake data.
	Args:
		fake_image_logits: the output of the discriminator for 
			fake images.

	Returns:
		The accuracy of the discriminator for fake images.'''

	pro_fake_logits = tf.equal(tf.round(fake_image_logits), tf.zeros_like(fake_image_logits))

	acc_fake = tf.reduce_mean(tf.cast(pro_fake_logits, tf.float32))

	return acc_fake