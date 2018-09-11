import moderngl
import os

from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
import numpy as np

vertex_shader = '''
               #version 330 core
               in vec3 in_vert;
                //out vec3 out_normal;

               void main() {
                   gl_Position = vec4(in_vert, 1.0);
               }
               '''
fragment_shader = '''
                #version 330 core

                uniform vec3 color;

                in vec3 out_normal;
                out vec4 f_color;

                const vec3 light1 = vec3(1, 1, -1);
                const vec3 light2 = vec3(0, 0, 1);

                const vec3 ambientLight = vec3(0.13, 0.13, 0.13);
                const float shininess = 16.0;
                void main() {
                    vec3 viewDir = vec3(0,0,1);
                    vec3 normal = normalize(out_normal);
                    float lambert1 = max(0, dot(normal, -normalize(light1)));
                    float lambert2 = max(0, dot(normal, -normalize(light2)));

                    vec3 halfDir = normalize(light2 + viewDir);
                    float specAngle = max(dot(halfDir, normal), 0.0);
                    float specular = pow(specAngle, shininess);


                    vec3 final_color = ambientLight + 
                                         0.8 * lambert1 * color + 
                                         1.0 * lambert2 * color;

                    f_color = vec4(final_color, 1.0);
                }
                '''
geometry_shader = '''
                #version 330 core

                layout(triangles) in;
                layout(triangle_strip, max_vertices = 3) out;

                out vec3 out_normal;

                void main() {
                    vec4 edge1 = gl_in[0].gl_Position - gl_in[1].gl_Position;
                    vec4 edge2 = gl_in[0].gl_Position - gl_in[2].gl_Position;
                    out_normal = cross(edge1.xyz, edge2.xyz);

                    for(int i = 0; i < 3; i++) {
                        gl_Position = gl_in[i].gl_Position;
                        EmitVertex();
                    }
                    EndPrimitive();
                }
                '''

ctx = moderngl.create_standalone_context()

ctx.enable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.CULL_FACE)


def render_model(verts, faces, image_size=256):
    prog = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        geometry_shader=geometry_shader
    )
    verts = verts * 10.

    xmin, xmax, xoffset = verts[:, 0].min(), verts[:, 0].max(), 0
    ymin, ymax, yoffset = verts[:, 1].min(), verts[:, 1].max(), 0
    zmin, zmax, zoffset = verts[:, 2].min(), verts[:, 2].max(), 0

    if xmax > 1: xoffset -= xmax - 1
    if xmin < -1: xoffset -= 1 + xmin

    if ymax > 1: yoffset -= ymax - 1
    if ymin < -1: yoffset -= 1 + ymin

    if zmax > 1: zoffset -= zmax - 1
    if zmin < -1: zoffset -= 1 + zmin

    verts[:, 0] += xoffset
    verts[:, 1] += yoffset
    verts[:, 2] += zoffset

    vboPos = ctx.buffer(verts.astype('f4').tobytes())

    ibo = ctx.buffer(faces.astype('i4').tobytes())

    vao_content = [
        # 3 floats are assigned to the 'in' variable named 'in_vert' in the shader code
        (vboPos, '3f', 'in_vert')
    ]

    vao = ctx.vertex_array(prog, vao_content, ibo)

    # Framebuffers
    fbo1 = ctx.framebuffer([ctx.renderbuffer((image_size, image_size), samples=8)])
    fbo2 = ctx.framebuffer([ctx.renderbuffer((image_size, image_size))])

    prog['color'].value = (1, 0, 0)

    # Rendering
    fbo1.use()
    ctx.clear(0.9, 0.9, 0.9)
    vao.render()

    # Downsampling and loading the image using Pillow
    ctx.copy_framebuffer(fbo2, fbo1)
    data = fbo2.read(components=3, alignment=1)
    img = Image.frombytes('RGB', fbo2.size, data).transpose(Image.FLIP_TOP_BOTTOM)

    prog.release()
    vboPos.release()
    ibo.release()
    vao.release()
    fbo1.release()
    fbo2.release()

    return img


if __name__ == '__main__':

    cols, rows = 16, 16
    width = 128
    result_length = cols * width

    res = Image.new("RGB", (int(result_length), int(result_length)))

    path = '/home/dawars/Downloads/handsOnly_SCANS/'
    for i, file in enumerate(sorted(os.listdir(path))[:rows * cols]):
        print(f"reading {file}")
        plydata = PlyData.read(os.path.join(path, file))

        faces = plydata['face'].data['vertex_indices'].tolist()
        faces = np.array(faces, dtype=np.int32)

        verts = plydata['vertex']
        verts = np.stack((verts['x'], verts['y'], verts['z']), axis=-1).reshape(-1, 3)

        x = i % cols
        y = i // cols

        print(f"{i}. Rendering at {x}, {y}")

        img = render_model(verts, faces, image_size=width)

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), file, fill=(0, 0, 0), font=ImageFont.truetype("arial"))

        x_pos = x * width
        y_pos = y * width

        res.paste(img, (int(x_pos), int(y_pos)))
    print("Manifold rendered")

    res.save("./mano_poses_high.png")
