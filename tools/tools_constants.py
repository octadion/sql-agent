# Retriever tool
few_shots_examples = {
    "Berikan daftar semua nama anggota di tulungagung.": """SELECT nama_gtk FROM anggota WHERE LOWER(kota) LIKE '%tulungagung%';""",
    "Berapa jumlah anggota yang berasal dari sekolah negeri?": """SELECT COUNT(*) FROM anggota WHERE is_negeri LIKE "NEGERI";""",
    "Siapa nama ketua komunitas dengan anggota terbanyak?": """SELECT nama_ketua AS ketua_komunitas, COUNT(anggota_id) AS jumlah_anggota FROM anggota GROUP BY nama_ketua ORDER BY jumlah_anggota DESC LIMIT 1;""",
    "Berapa jumlah anggota yang aktif pada tahun 2023?": """SELECT COUNT(*) FROM anggota WHERE status_aktif = 'Aktif' AND tahun_terdaftar = 2023;""",
    "Urutkan nama anggota secara alfabetis di jember.": """SELECT nama_gtk FROM anggota WHERE LOWER(kota) LIKE '%jember%' ORDER BY nama_gtk;""",
    "Apakah ada anggota yang berasal dari provinsi Bali?": """SELECT * FROM anggota WHERE LOWER(provinsi) LIKE LOWER('%bali%');""",
    "Berapa jumlah anggota yang telah sertifikasi?": """SELECT COUNT(*) AS jumlah_anggota_sertifikasi FROM anggota WHERE status_sertifikasi LIKE 'Sudah';""",
    "Siapa anggota tertua dalam komunitas?": """SELECT nama_gtk, umur FROM anggota ORDER BY umur DESC LIMIT 1;""",
    "Berapa jumlah anggota yang memiliki status perkawinan 'Kawin'?": """SELECT COUNT(*) FROM anggota WHERE LOWER(status_perkawinan) LIKE LOWER('%kawin%');""",
    "Apakah ada anggota yang berusia di atas 50 tahun?": """SELECT nama_gtk, umur FROM anggota WHERE umur > 50;""",
    "Urutkan nama anggota berdasarkan usia dari yang tertua ke yang termuda di probolinggo.": """SELECT nama_gtk, umur FROM anggota WHERE LOWER(kota) LIKE '%probolinggo%' ORDER BY umur DESC;""",
    "Berapa jumlah anggota yang memiliki status keaktifan 'Aktif'?": """SELECT COUNT(*) FROM anggota WHERE LOWER(status_keaktifan) LIKE LOWER('%aktif%');""",
    "Siapa nama ketua komunitas dengan nomor SK '800/2830/408.37.06/2008'?": """SELECT nama_ketua FROM anggota WHERE no_sk = '800/2830/408.37.06/2008';""",
    "Berikan daftar nama anggota yang berjenis kelamin perempuan di pacitan.": """SELECT nama_gtk FROM anggota WHERE LOWER(kelamin) LIKE '%p%' AND LOWER(kota) LIKE LOWER('%pacitan%');""",
    "Apa usia terkecil dari anggota yang memiliki status keaktifan di sekolah 'Aktif'?": """SELECT MIN(umur) FROM anggota WHERE LOWER(status_keaktifan) LIKE '%aktif%';""",
    "Apa usia terbesar dari anggota yang berusia di bawah 30 tahun?": """SELECT MAX(umur) FROM anggota WHERE umur < 30;""",
    "Urutkan nama anggota berdasarkan tipe komunitas secara alfabetis di bojonegoro.": """SELECT nama_gtk, tipe_komunitas FROM anggota WHERE LOWER(kota) LIKE '%bojonegoro%' ORDER BY tipe_komunitas ASC;""",
    "Siapa nama ketua komunitas dengan anggota termuda di banyuwangi?": """SELECT nama_ketua FROM anggota WHERE LOWER(kota) LIKE '%banyuwangi%' ORDER BY umur ASC LIMIT 1;""",
    "Berapa rata-rata umur semua anggota yang memiliki kualifikasi pendidikan S2": """SELECT AVG(umur) FROM anggota WHERE LOWER(kualifikasi) LIKE '%s2%';"""
}

retriever_tool_description = (
    "The 'sql_get_few_shot' tool is designed for efficient and accurate retrieval of "
    "SQL query examples closely related to a given user query. It identifies the most "
    "relevant pre-defined SQL query from a curated set."
)

# Other tools

COLUMNS_DESCRIPTIONS = {
    "uuid": "Identifier unik untuk setiap entitas dalam dataset (contoh: 3273e217-f28b-4d2c-9593-8ea97d1f6f32)",
    "captured_at": "Waktu atau timestamp ketika data diambil (contoh: \"8326858679\")",
    "anggota_id": "ID anggota dalam format float64 (contoh: .nan, 1412827.0)",
    "nopes": "Nomor Peserta (contoh: \"5003658808\")",
    "nuptk": "Nomor Unik Pendidik dan Tenaga Kependidikan dalam format float64 (contoh: .nan, 36761664200013.0)",
    "nuks": "Nomor Unik Komunitas Sekolah (contoh: \"\", \"19023L0010503232056102\")",
    "email_login": "Alamat email untuk login (contoh: g-cxt41@yahoo.com)",
    "nama_gtk": "Nama guru atau tenaga kependidikan (contoh: ISMIATI, YULITA DEWI PUSPARINA)",
    "foto_ptk": "Nomor identifikasi untuk foto guru atau tenaga kependidikan (contoh: \"8436182719\")",
    "asal_sekolah": "Nama sekolah asal guru atau tenaga kependidikan (contoh: SD NEGERI KAWU 4)",
    "status_satminkal": "Status keanggotaan dalam Satminkal (contoh: Satminkal, Bukan Satminkal)",
    "nama_komunitas": "Nama komunitas atau gugus dalam format teks (contoh: \"\", \"MGMP SMK - Kimia - SMK (Produktif) - SMK (Produktif)\")",
    "no_sk": "Nomor Surat Keputusan (contoh: \"800/2830/408.37.06/2008\")",
    "tgl_sk": "Tanggal Surat Keputusan dalam format YYYY-MM-DD (contoh: \"2017-06-21\")",
    "tipe_komunitas": "Tipe komunitas (contoh: KKKS SD, KKG SLB, MGMP SMA)",
    "mapel_ukg": "Mata pelajaran yang diambil dalam UKG (contoh: Seni Budaya Tari, Bahasa Inggris)",
    "jenjang": "Jenjang pendidikan (contoh: SMA, SMP, SMK)",
    "nopes_ketua": "Nomor Peserta Ketua Komunitas dalam format float64 (contoh: .nan, 201511032620.0)",
    "nama_ketua": "Nama ketua komunitas (contoh: MOH. JURI)",
    "alamat": "Alamat rumah atau tempat tinggal (contoh: Rt 08 Rw 02, Jati)",
    "kecamatan": "Nama kecamatan tempat tinggal (contoh: Bangorejo)",
    "kota": "Nama kota atau kabupaten tempat tinggal (contoh: Kota Mojokerto)",
    "provinsi": "Nama provinsi tempat tinggal (contoh: Jawa Timur)",
    "komunitas_id": "ID komunitas dalam format float64 (contoh: 38926.0)",
    "is_negeri": "Status apakah anggota berasal dari sekolah negeri atau swasta (contoh: NEGERI, SWASTA)",
    "status_anggota": "Status keanggotaan (contoh: Belum Tergabung, Tergabung)",
    "jabatan_komunitas": "Jabatan dalam komunitas (contoh: Bendahara Komunitas, Ketua Komunitas)",
    "kelamin": "Jenis kelamin (contoh: P, L)",
    "umur": "Usia anggota dalam format int64 (contoh: 34)",
    "simpkb_email": "Alamat email untuk SIM PKB (contoh: 855ql842n6@gmail.com)",
    "simpkb_no_hp": "Nomor handphone untuk SIM PKB (contoh: \"7020344991\")",
    "dapodik_email": "Alamat email untuk Dapodik (contoh: zxcr6qlj@gmail.com)",
    "dapodik_no_hp": "Nomor handphone untuk Dapodik dalam format float64 (contoh: .nan, 81230207872.0)",
    "mapel_ukg_ptk": "Mata pelajaran yang diambil dalam UKG untuk PTK (contoh: Matematika, Akomodasi Perhotelan)",
    "tugas": "Jabatan atau tugas anggota (contoh: PLT Kepala Sekolah, Kepala Sekolah)",
    "jenis_ptk": "Jenis PTK (contoh: Guru Mapel, Guru TIK)",
    "kualifikasi": "Kualifikasi pendidikan (contoh: S2, D1, Paket C)",
    "pegawai": "Status kepegawaian (contoh: Tenaga Honor Sekolah)",
    "golongan": "Golongan pegawai (contoh: IVb, IX, III/d)",
    "alamat_sekolah": "Alamat sekolah tempat anggota bekerja (contoh: Alastlogo)",
    "dapodik_ptk_id": "ID Dapodik untuk PTK (contoh: 8ADAEAFC-7B21-E211-8821-31304FD3D4E3)",
    "longitude": "Koordinat longitude dalam format float64 (contoh: .nan, 111.1871)",
    "latitude": "Koordinat latitude dalam format float64 (contoh: .nan, -8.126610205563)",
    "no_kk": "Nomor Kartu Keluarga dalam format float64 (contoh: 3507141601200008.0)",
    "nik": "Nomor Induk Kependudukan (contoh: 0289 0306 0743 2412)",
    "nip": "Nomor Induk Pegawai (contoh: \"196607082001011021\", \"-\", \"198908012022212008\")",
    "dapodik_no_hp_operator": "Nomor handphone operator untuk Dapodik (contoh: Kartu Matrix)",
    "nama_ibu_kandung": "Nama ibu kandung anggota (contoh: Wardah)",
    "npwp": "Nomor Pokok Wajib Pajak (contoh: \"921195939647000\")",
    "nama_npwp": "Nama pemilik NPWP (contoh: IMRO\"ATU SHOLIHAH)",
    "domisili_alamat_jalan": "Alamat jalan tempat tinggal anggota (contoh: Lumutan)",
    "domisili_nama_dusun": "Nama dusun tempat tinggal anggota (contoh: Winong)",
    "domisili_kode_pos": "Kode pos tempat tinggal anggota dalam format float64 (contoh: 63173.0)",
    "domisili_rt": "Nomor RT tempat tinggal anggota dalam format float64 (contoh: .nan, 38.0)",
    "domisili_rw": "Nomor RW tempat tinggal anggota dalam format float64 (contoh: 22.0)",
    "domisili_desa_kelurahan": "Nama desa atau kelurahan tempat tinggal anggota (contoh: GONDANG)",
    "domisili_kecamatan": "Nama kecamatan tempat tinggal anggota (contoh: Kec. Dander)",
    "domisili_kota_kabupaten": "Nama kota atau kabupaten tempat tinggal anggota (contoh: Kab. Pasuruan)",
    "domisili_provinsi": "Nama provinsi tempat tinggal anggota (contoh: Prov. Jawa Timur)",
    "status_keaktifan": "Status keaktifan dari Dapodik (contoh: Aktif)",
    "agama": "Agama anggota (contoh: Katholik, Islam)",
    "status_perkawinan": "Status perkawinan anggota (contoh: Kawin, Janda/Duda)",
    "kewarganegaraan": "Kewarganegaraan anggota (contoh: Indonesia)",
    "niy_nigk": "Nomor Induk Yayasan/Nomor Induk Guru dan Karyawan (contoh: \"9524778717\")",
    "npsn": "Nomor Pokok Sekolah Nasional dalam format int64 (contoh: 20507662)",
    "sekolah_id": "ID sekolah dalam format int64 (contoh: 20501574)",
    "instansi_id": "ID instansi (contoh: \"\")",
    "dapodik_sekolah_id": "ID Dapodik sekolah (contoh: C080CBF0-8B18-E111-B6FA-B72E8DC4E109)",
    "is_aktif_sekolah": "Status aktif atau tidaknya sekolah dalam format int64 (contoh: 1)",
    "is_aktif": "Status aktif atau tidaknya anggota dalam format int64 (contoh: 1)",
    "nrg": "Nomor Registrasi Guru dalam format object (contoh: \"8612637719\")",
    "thn_sertifikasi": "Tahun sertifikasi anggota dalam format float64 (contoh: 2023.0)",
    "mapel_sertifikasi": "Mapel yang disertifikasi oleh anggota (contoh: \"[2017-187] KIMIA\")",
    "last_access": "Waktu terakhir akses anggota dalam format object (contoh: \"2024-02-12 04:22:29\")",
    "status_aktif": "Status aktif atau tidaknya anggota (contoh: Aktif)",
    "status_sinkron": "Status sinkronisasi data (contoh: Sinkron)",
    "jenis_ptk_kelompok": "Jenis PTK kelompok (contoh: Pengajar/Guru)",
    "mapel_ajar_dapodik": "Mata pelajaran yang diajar dalam Dapodik (contoh: Alat Industri Kimia)",
    "k_kota": "Kode Kota dalam format int64 (contoh: 3524)",
    "instansi_tugas": "Instansi atau sekolah tempat anggota bertugas (contoh: \"\", \"SMP ISLAM MIFTAHUSSURUR\")",
    "instansi_npsn_tugas": "Nomor Pokok Sekolah Nasional tempat anggota bertugas dalam format float64 (contoh: 20515030.0)",
    "paspor_id": "Nomor paspor anggota dalam format int64 (contoh: 1800011)",
    "bentuk_pendidikan": "Bentuk pendidikan tempat anggota bekerja (contoh: KB, SD)",
    "sumber_data": "Sumber data yang digunakan untuk mengisi dataset (contoh: Migrasi/Dapodik)",
    "kota_status_3t": "Status 3T kota tempat tinggal anggota (contoh: \"-\")",
    "jenis_keluar": "Jenis keluar anggota dari komunitas (contoh: Aktif, Alih Fungsi)",
    "tmp_lahir": "Tempat lahir anggota (contoh: GRESIK)",
    "tgl_lahir": "Tanggal lahir anggota dalam format object (contoh: \"1988-07-27\")",
    "status_aktif_sekolah": "Status aktif Sekolah (contoh: Ya)",
    "sektor_vokasi": "Sektor vokasi atau bidang pekerjaan (contoh: Hospitality)",
    "naungan": "Naungan atau lembaga yang menjadi otoritas anggota (contoh: Kemendikbud)",
    "sekolah_akreditasi": "Status akreditasi sekolah (contoh: A)",
    "lembaga_akreditasi": "Lembaga akreditasi yang memberikan akreditasi (contoh: BAN-SM)",
    "tahun_terdaftar": "Tahun terdaftar anggota dalam format int64 (contoh: 2023)",
    "asal_sekolah_jenis_naungan": "Jenis naungan sekolah asal anggota (contoh: Dinas Pendidikan Kabupaten/Kota)",
    "asal_sekolah_naungan": "Naungan sekolah asal anggota (contoh: Dinas Pendidikan Kab. Ponorogo)",
    "status_sertifikasi": "Status sertifikasi anggota (contoh: Sudah, Belum)",
    "status_nuptk": "Status Nuptk anggota (contoh: Sudah, Belum)"
}