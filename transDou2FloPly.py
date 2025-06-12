import os

def fix_ply_header(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if 'property double x' in line:
            new_lines.append('property float x\n')
        elif 'property double y' in line:
            new_lines.append('property float y\n')
        elif 'property double z' in line:
            new_lines.append('property float z\n')
        else:
            new_lines.append(line)

        if 'end_header' in line:
            break

    # Append point data
    point_data_start = len(new_lines)
    new_lines += lines[point_data_start:]

    with open(file_path, 'w') as f:
        f.writelines(new_lines)

# 替换该路径为你的 ply 文件目录
ply_dir = '/home/zzw/YOPO/run/yopo_gazebo/'
for fname in os.listdir(ply_dir):
    if fname.endswith('.ply'):
        print("Converting2Float:", fname)
        fix_ply_header(os.path.join(ply_dir, fname))

