"""
Functions for rendering a single MANO model to image and manifold
"""
import moderngl
import os

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch

from utils.mano_utils import *

vertex_shader = '''
               #version 330
               in vec3 in_vert;
               in vec3 in_normal;
               out vec3 out_normal;
               void main() {
                   gl_Position = vec4(in_vert, 1.0);
                   out_normal = in_normal;
               }
               '''
fragment_shader = '''
                   #version 330
                   in vec3 out_normal;
                   out vec4 f_color;

                   const vec3 lightDir = vec3(1, 1, 1);
                   void main() {
                       vec3 light = -normalize(lightDir);

                       vec3 normal = normalize(out_normal);
                       float lambert = min(0, dot(normal, light));

                       vec3 color = lambert * vec3(1.0, 0.0, 0.0);
                       f_color = vec4(color, 1.0);
                   }
                   '''


class HandRenderer:
    def __init__(self, image_size=128, sess=None):
        """
        Class for rendering a hand from parameters or manifold
        :param sess:
        :param image_size: size of a single hand image
        """
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.Session()

        self.image_size = image_size

        # graphics
        self.ctx = moderngl.create_standalone_context()

        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        self.vbo = self.ctx.buffer(reserve=778 * 3 * 4, dynamic=True)
        self.ibo = self.ctx.buffer(get_mano_faces().astype('i4').tobytes())

        vao_content = [
            # 3 floats are assigned to the 'in' variable named 'in_vert' in the shader code
            (self.vbo, '3f', 'in_vert')
        ]

        self.vao = self.ctx.vertex_array(self.prog, vao_content, self.ibo)

        # Framebuffers

        self.fbo1 = self.ctx.framebuffer(self.ctx.renderbuffer((image_size, image_size), samples=1))
        self.fbo2 = self.ctx.framebuffer(self.ctx.renderbuffer((image_size, image_size)))

    def __del__(self):
        self.sess.close()

        self.prog.release()
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.fbo1.release()
        self.fbo2.release()

    def render_manifold(self, decoder, name="./manifold.png", bounds=(-4, 4), steps=0.5, verbose=False):
        """
        Render a 2D posed hand manifold
        :param decoder: pytorch decoder function 2 -> 45 params
        :param name: filename
        :param bounds: bounds of the sampling along the x and y axis
        :param steps: step size for sampling between the bounds
        :param verbose: print progress
        :returns rendered image
        """

        os.makedirs(os.path.dirname(name), exist_ok=True)

        result_length = self.image_size * (bounds[1] - bounds[0]) / steps

        # coordinates to sample at
        sampling_grid = np.mgrid[bounds[0]:bounds[1]:steps, bounds[0]:bounds[1]:steps]
        encoded = sampling_grid.reshape(2, -1).T

        _, cols, rows = sampling_grid.shape

        encoded = torch.tensor(encoded, dtype=torch.float).cuda()
        batch_size = len(encoded)

        rot = np.zeros([batch_size, 3])
        shape = np.zeros([batch_size, 10])

        decoded_poses = decoder(encoded)
        decoded_poses = decoded_poses.cpu().detach().numpy() + mano_data['hands_mean']

        decoded_poses = np.concatenate((rot, decoded_poses), axis=1)
        vertices = get_mano_vertices(shape, decoded_poses, sess=self.sess)

        res = Image.new("RGB", (int(result_length), int(result_length)))
        for x in range(cols):
            for y in range(rows):
                if verbose:
                    print("Rendering at {x}, {y}".format(x=x, y=y))

                model_index = y * rows + x

                img = self.render_mano(vertices[model_index])
                # mano_to_OBJ(shape, decoded_poses, "./test.obj")

                x_pos = x * self.image_size
                y_pose = y * self.image_size

                res.paste(img, (int(x_pos), int(y_pose)))
        if verbose:
            print("Manifold rendered")
        res.save(name)
        return res

    def render_mano(self, mano_vertices):
        """
        Render Mano on a single image
        :param mano_vertices: vertices of model render
        :return image of the hand
        """
        vertices = mano_vertices * 10.

        # todo calculate normals
        # https://gamedev.stackexchange.com/questions/152991/how-can-i-calculate-normals-using-a-vertex-and-index-buffer
        # http://www.opengl-tutorial.org/beginners-tutorials/tutorial-8-basic-shading/#triangle-normals

        self.vbo.write(vertices.astype('f4').tobytes())

        # Rendering
        self.fbo1.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.9, 0.9, 0.9)
        self.vao.render()

        # Downsampling and loading the image using Pillow

        self.ctx.copy_framebuffer(self.fbo2, self.fbo1)
        data = self.fbo2.read(components=3, alignment=1)
        img = Image.frombytes('RGB', self.fbo2.size, data).transpose(Image.FLIP_TOP_BOTTOM)

        # img.show()
        return img


if __name__ == '__main__':
    renderer = HandRenderer(128)

    # rendering mano
    # over ssh run with alias xpy='xvfb-run -s "-screen 0 1x1x24" python3'
    # xpy -m ModernGL
    # source https://github.com/cprogrammer1994/ModernGL/tree/master/examples
    pose = np.zeros([1, 48])
    finger = 0
    joint = 0
    i = 3 + 3 * finger + joint
    pose[0, i + 1] = np.pi / 2

    from block_timer.timer import Timer

    with Timer():
        poses = get_mano_vertices(np.zeros([1, 10]), pose)

    img = renderer.render_mano(poses[0])

    plt.imsave('rendering_test.png', img)

    # rendering manifold
    from pose_autoencoders.vanilla_ae import autoencoder

    ae = autoencoder().cuda()  # Load a premade autoencoder
    ae.load_state_dict(torch.load('../pose_autoencoders/sim_autoencoder.pth'))

    import time

    start = time.time()
    renderer.render_manifold(ae.decoder, './manifold_test.png', verbose=False)
    end = time.time()

    elapsed = end - start
    print(elapsed)
