import os
import sys
import tensorflow as tf
from utils import *
import numpy as np
import matplotlib.image as img 
import cv2

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
sess = tf.InteractiveSession()

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[m, -1, n_C]),perm=[0,2,1])
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[m, -1, n_C]),perm=[0,2,1])
    J_content = tf.reduce_sum((a_C_unrolled-a_G_unrolled)**2)/(4*n_H* n_W* n_C)
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S,(-1,n_C)))
    a_G = tf.transpose(tf.reshape(a_G,(-1,n_C)))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = tf.reduce_sum((GG-GS)**2)/(2*n_H*n_W*n_C)**2    
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]
def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 50, beta = 50):
    J = alpha*J_content+beta*J_style
    return J

content_image = cv2.imread("images/mando.jpg")
b,g,r=cv2.split(cv2.resize(content_image,(400,300)))
content_image=cv2.merge([r,g,b])
content_image = reshape_and_normalize_image(content_image)

style_image = cv2.imread("images/design.jpg")
b,g,r=cv2.split(cv2.resize(style_image,(400,300)))
style_image=cv2.merge([r,g,b])
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
b,g,r=cv2.split(generated_image[0])
img=cv2.merge([r,g,b])
cv2.imwrite("output/noise.jpg", np.clip(img, 0, 255).astype('uint8'))
        
        
        
def model_nn(sess, input_image, num_iterations = 200):
    sess.run(tf.global_variables_initializer())
    
    generated_image= sess.run(model["input"].assign(input_image))  
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model["input"])
        
        if i%20 == 0:
            
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            b,g,r=cv2.split(generated_image[0])
            img=cv2.merge([r,g,b])
            cv2.imwrite("output/" + str(i) + ".jpg", np.clip(img, 0, 255).astype('uint8'))
    b,g,r=cv2.split(generated_image[0])
    img=cv2.merge([r,g,b])
    cv2.imwrite('output/generated_image.jpg', np.clip(img, 0, 255).astype('uint8'))

content_path=input("enter the name of Content image: ")
content_image = cv2.imread("images/"+content_path)
b,g,r=cv2.split(cv2.resize(content_image,(400,300)))
content_image=cv2.merge([r,g,b])
content_image = reshape_and_normalize_image(content_image)

style_path=input("enter the name of Style image: ")
style_image = cv2.imread("images/"+style_path)
b,g,r=cv2.split(cv2.resize(style_image,(400,300)))
style_image=cv2.merge([r,g,b])
style_image = reshape_and_normalize_image(style_image)


sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out

J_content = compute_content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

alp=float(input("enter the percentage of content image in final: "))
J = total_cost(J_content, J_style, alpha = alp, beta = 100-alp)
optimizer = tf.train.AdamOptimizer(2.0)

train_step = optimizer.minimize(J)


itter=int(input("no of iterrations: "))

model_nn(sess, generated_image,itter)


