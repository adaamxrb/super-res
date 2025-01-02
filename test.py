from __future__ import print_function
import argparse
import os
import torch
import time
import cv2
from torch.utils.data import DataLoader
from dbpn import Network as DBPN
from data import get_eval_set
from functools import reduce
from tabulate import tabulate

# ============================
# 🎯 Konfigurasi Pengujian
# ============================
parser = argparse.ArgumentParser(description='Contoh Penggunaan :\n python test.py --upscale_factor 2...')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int,default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--output', default='Hasil/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='Set5_LR_x4')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--model', default=r'tr_model/DIV2K_train_HR-DBPN-x4-residual-epoch-99.pth')

opt = parser.parse_args()
gpus_list = range(opt.gpus)

# ============================
# 🚦 GPU Check
# ============================
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception(
        "tidak ada GPU yang ditemukan, jalankan ulang dengan konfigurasi --gpu_mode False")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

# ============================
# 📊 Menampilkan Konfigurasi yang digunakan
# ============================
os.system('clear')
print("\n⚙️  KONFIGURASI PENGUJIAN")
print("=" * 80)
for arg in vars(opt):
    print(f"{arg.replace('_', ' ').capitalize():<25}: {getattr(opt, arg)}")
print("-" * 80)

# ============================
# 📂 Menyiapkan Dataset
# ============================
print("📚 ==> Loading dataset...")
test_set = get_eval_set(os.path.join(
    opt.input_dir, opt.test_dataset), opt.upscale_factor)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

# ============================
# 🛠️ Menyiapkan Model
# ============================
print("🛠️  ==> Memuat model...")
model = DBPN(num_channels=3, base_filter=64, feat=256,
             num_stages=7, scale_factor=opt.upscale_factor)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(
    opt.model, map_location=lambda storage, loc: storage, weights_only=True))
print("✅ Trained Super Resolution model berhasil dimuat.")

if cuda:
    print("-" * 80)
    print("🖥️  GPU Mode : ACTIVE")
    print(f"🖥️  GPU      : {torch.cuda.get_device_name(gpus_list[0])}")
    print("=" * 80)

    model = model.cuda(gpus_list[0])


# ============================
# 🧠 Fungsi Pengujian
# ============================
def eval():
    print("\n🧪 PENGUJIAN DIMULAI")
    print("=" * 48)

    results = []
    model.eval()
    for idx, batch in enumerate(testing_data_loader, 1):
        try:
            with torch.no_grad():
                input, bicubic, name = batch[0], batch[1], batch[2]
            if cuda:
                input = input.cuda(gpus_list[0])
                bicubic = bicubic.cuda(gpus_list[0])

            t0 = time.time()
            if opt.chop_forward:
                with torch.no_grad():
                    prediction = chop_forward(input, model, opt.upscale_factor)
            else:
                if opt.self_ensemble:
                    with torch.no_grad():
                        prediction = x8_forward(input, model)
                else:
                    with torch.no_grad():
                        prediction = model(input)

            if opt.residual:
                prediction = prediction + bicubic

            t1 = time.time()
            save_path = save_img(prediction.cpu().data, name[0])
            results.append(
                [idx, name[0], f"{t1 - t0:.4f} dtk", f"📁 {save_path}", "✅ Sukses"])
            print(f"==> Memproses Gambar({idx}/{len(testing_data_loader)}) || Waktu: {t1 - t0:.4f} dtk.")
        except Exception as e:
            results.append([idx, name[0], "❌ Error", "-", f"❌ {e}"])

    print("=" * 48)
    print("\n📊 HASIL PENGUJIAN:")
    print(tabulate(results, headers=["No", "Gambar", "Waktu Proses", "Saved Path", "Status"], tablefmt="grid"))

# ============================
# 💾 Fungsi Menyimpan Gambar
# ============================


def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    save_dir = os.path.join(opt.output, opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = os.path.join(save_dir, f'DBPN_{img_name}')
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return save_fn

# ============================
# 🔄 Fungsi x8 Forward
# ============================


def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single':
            v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')

    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output

# ============================
# ✂️ Fungsi Chop Forward
# ============================


def chop_forward(x, model, scale, shave=8, min_size=80000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            if opt.self_ensemble:
                with torch.no_grad():
                    output_batch = x8_forward(input_batch, model)
            else:
                with torch.no_grad():
                    output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs)
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = outputlist[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


# ============================
# 🚀 Main Entry Point
# ============================
if __name__ == '__main__':
    eval()
    print("\n✅ Pengujian Selesai, Hasil disimpan di:", opt.output, opt.test_dataset)
