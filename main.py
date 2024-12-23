from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from data import get_training_set
import time

# ============================
# 🎯 Konfigurasi Pelatihan
# ============================
parser = argparse.ArgumentParser(description='Contoh Penggunaan : \npython main.py --upscale_factor 2 --batchSize 4 --epochs 200 --snapshots 10 --start_iter 1 --lr 0.01 --gpu_mode True --threads 4 --seed 123 --gpus 1 --data_dir ./Dataset --data_augmentation True --hr_train_dataset DIV2K_train_HR --residual True --patch_size 40 --pretrained_sr pt_model/DBPN_2x.pth --pretrained True --save_folder tr_model/ --prefix -x2-residual')
parser.add_argument('--upscale_factor', type=int, default=4, help="Faktor upscaling")
parser.add_argument('--batchSize', type=int, default=2, help='Ukuran batch')
parser.add_argument('--epochs', type=int, default=50, help='Jumlah epoch')
parser.add_argument('--snapshots', type=int, default=10, help='Interval Checkpoint')
parser.add_argument('--start_iter', type=int, default=1, help='Mulai iterasi')
parser.add_argument('--lr', type=float, default=1e-4, help='Jumlah Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True, help='Apakah menggunakan GPU')
parser.add_argument('--threads', type=int, default=4, help='Jumlah thread data loader')
parser.add_argument('--seed', type=int, default=123, help='Random seed yang digunakan')
parser.add_argument('--gpus', default=1, type=int, help='Jumlah gpu yang digunakan')
parser.add_argument('--data_dir', type=str, default='./Dataset' , help='lokasi dataset')
parser.add_argument('--data_augmentation', type=bool, default=True, help='Apakah menggunakan data augmentation')
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR', help='Nama dataset high resolution')
parser.add_argument('--residual', type=bool, default=True, help='Apakah menggunakan residual learning')
parser.add_argument('--patch_size', type=int, default=40,help='Ukuran gambar high resolution yang di-crop')
parser.add_argument('--pretrained_sr', default='pt_model/DBPN_4x.pth', help='Lokasi pretrained model base sr')
parser.add_argument('--pretrained', type=bool, default=True, help='Apakah menggunakan pretrained model')
parser.add_argument('--save_folder', default='tr_model/', help='Lokasi penyimpanan model checkpoint')
parser.add_argument('--prefix', default='-x4-residual-dbpn', help='Jenis model yang disimpan')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True

# ============================
#  🧪 Fungsi Pelatihan
# ============================


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = batch[0], batch[1], batch[2]
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("==> Epoch[{}]({}/{})  : 🔄 Loss: {:.4f} || ⏱  Waktu: {:.4f} dtk.".format(
            epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    avg_loss = epoch_loss / len(training_data_loader)
    return avg_loss

# ============================
# 🧪 Fungsi Pengujian
# ============================


def test(epoch):
    print("=" * 68)
    print("\n🧪 Pengujian Epoch {}".format(epoch))
    print("=" * 68)
    avg_psnr = 0
    for iteration, batch in enumerate(testing_data_loader, 1):
        input, target, bicubic = batch[0], batch[1], batch[2]
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        prediction = model(input)
        if opt.residual:
            prediction = prediction + bicubic

        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr

        print("===> Melakukan Tes [{}]({}/{})  : 🎯  PSNR: {:.4f} dB".format(
            epoch, iteration, len(testing_data_loader), psnr))

    avg_psnr = avg_psnr / len(testing_data_loader)
    return avg_psnr

# ============================
# 💾 Fungsi Checkpoint
# ============================


def checkpoint(epoch):
    model_out_path = opt.save_folder + opt.hr_train_dataset + "-" + \
        "DBPN" + opt.prefix + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint disimpan di: {}".format(model_out_path))


if __name__ == '__main__':
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception(
            "Tidak ada GPU yang ditemukan, jalankan ulang dengan konfigurasi --cuda False")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    os.system('clear')
    print("\n⚙️ KONFIGURASI MODEL TRAINING")
    print("=" * 68)

    for arg in vars(opt):
        print(f"{arg.replace('_', ' ').capitalize():<25}: {getattr(opt, arg)}")
    print("-" * 68)

    if cuda:
        print("🖥️ GPU Mode   : ACTIVE")
        print(f"🖥️ GPU        : {torch.cuda.get_device_name(gpus_list[0])}")
        print("-" * 68)
    else:
        print("🖥️ GPU Mode : INACTIVE")
        print(f"🖥️ Menggunakan CPU")
        print("-" * 68)

    print("📂 Memuat dataset...")
    train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
    training_data_loader = DataLoader(
        dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("🛠️  Menyiapkan model...")
    print("🔍 Memuat dataset penguji...")
    test_set = get_training_set(opt.data_dir, opt.hr_train_dataset,
                                opt.upscale_factor, opt.patch_size, opt.data_augmentation)
    testing_data_loader = DataLoader(
        dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    model = DBPN(num_channels=3, base_filter=64, feat=256, num_stages=7, scale_factor=opt.upscale_factor)

    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = nn.L1Loss()

    print("✅ Inisialisasi berhasil, Model Training akan segera dimulai")

    if opt.pretrained:
        model_name = os.path.join(opt.save_folder, opt.pretrained_sr)
        if os.path.exists(model_name):
            model.load_state_dict(torch.load(
                model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model berhasil dimuat.')

    if cuda:
        model = model.cuda(gpus_list[0])
        criterion = criterion.cuda(gpus_list[0])

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    start_time = time.time()
    prev_avg_loss = None
    prev_avg_psnr = None
    for epoch in range(opt.start_iter, opt.epochs + 1):
        print(
            "\n⚡ MEMULAI EPOCH {}".format(epoch))
        print("=" * 68)
        avg_loss = train(epoch)
        avg_psnr = test(epoch)
        elapsed_time = time.time() - start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        print("\n🚀 TRAINING PROGRESS - EPOCH {}".format(epoch))
        print("=" * 58)
        print("📊 Average Loss              : {:.4f}".format(avg_loss))
        if prev_avg_loss is not None:
            print("📊 Loss Change               : {:.4f}".format(avg_loss - prev_avg_loss))
        print("🎯 Average PSNR              : {:.4f} dB".format(avg_psnr))
        if prev_avg_psnr is not None:
            print("🎯 PSNR Change               : {:.4f} dB".format( - prev_avg_psnr))
        print("🔄 Total Epochs Run          : {}/{}".format(epoch, opt.epochs))
        print("🕒 Total Time Elapsed        : {}".format(elapsed_time_str))
        print("💡 Status                    : Training Berhasil ")
        print("=" * 58)

        prev_avg_loss = avg_loss
        prev_avg_psnr = avg_psnr

        if (epoch+1) % (opt.epochs/2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate diturunkan: lr={}'.format(
                optimizer.param_groups[0]['lr']))

        if (epoch+1) % (opt.snapshots) == 0:
            checkpoint(epoch)
