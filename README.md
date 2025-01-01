# Super-Resolution with Deep Back-Projection Networks

Repository ini merupakan implementasi dari jurnal berjudul "Deep Back-Projection Networks For Super-Resolution" yang ditulis oleh Haris, Muhammad A., Greg Shakhnarovich, dan Norimichi Ukita. Program ini menggunakan model Deep Back-Projection Networks (DBPN) untuk meningkatkan resolusi gambar.

## Deep Back-Projection Networks (DBPN)

![DBPN Model](assets/dbpn.png)

DBPN adalah model jaringan saraf tiruan yang digunakan untuk meningkatkan resolusi gambar. Model ini terdiri dari dua bagian utama, yaitu jaringan dekonvolusi dan jaringan proyeksi balik. Jaringan dekonvolusi bertugas untuk menghasilkan gambar dengan resolusi yang lebih tinggi, sedangkan jaringan proyeksi balik bertugas untuk memperbaiki gambar hasil dekonvolusi agar lebih mirip dengan gambar asli.

## Dataset

Dataset yang digunakan dalam training model adalah [DIV2K Dataset](https://figshare.com/articles/dataset/DIV2K_train_HR_zip/9785438/1?file=17544995), yang berisi gambar dengan resolusi tinggi dan rendah. Dataset ini sering digunakan dalam penelitian super-resolusi gambar.

## Result

Berikut adalah contoh hasil pengujian menggunakan dataset [Set5](https://paperswithcode.com/dataset/set5) dengan scale x2, x4, dan x8 menggunakan model DBPN yang sudah dilatih.

| Scale Factor | Input Image                    | Output Image                    |
| ------------ | ------------------------------ | ------------------------------- |
| <h3>x2</h3>  | ![Input Image](assets/i2x.png) | ![Output Image](assets/o2x.png) |
| <h3>x4</h3>  | ![Input Image](assets/i4x.png) | ![Output Image](assets/o4x.png) |
| <h3>x8</h3>  | coming soon                    | coming soon                     |

## Team

-   [Adham Roy Bhafiel](https://github.com/adaamxrb)
-   [Erika Putri Lestari](https://github.com/erikaprls)
-   [Azila Lailannafisa](https://github.com/azilafisa)
-   [Athaya Aqilah](https://github.com/athayaaqilaa)
-   [Bintang Tiara Pramesti](https://github.com/bintangtiara/)
