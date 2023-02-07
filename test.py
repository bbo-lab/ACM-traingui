import imageio.v3 as iio
import hashlib

file_name = "/media/smb/soma-fs.ad01.caesar.de/bbo/analysis/pose/data/user/Kay/../../../data/mp4/20220106_imutest-teensy_0011/360_0011.MP4"

plugin = "pyav"
reader = iio.imopen(file_name, "r", plugin=plugin)
img = reader.read(index=0)
print(hashlib.md5(img).hexdigest())
reader.read(index=0)
