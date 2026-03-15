import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# 1. VERİ OKUMA

# Features.npz dosyasını oku
data = np.load('Features.npz')
Features = data['Features']
# İlk 100 kişiyi al
Features = Features[:, :100, :]

# 2. NORMALİZASYON [0,1] ARALIĞINA

# Tüm verileri düzleştirerek min-max bul
all_data = Features.reshape(-1, Features.shape[-1])  # (1000, 6)
data_min = all_data.min(axis=0)
data_max = all_data.max(axis=0)

# [0,1] aralığına normalize et: (x - min) / (max - min)
# Her özellik (feature) için ayrı min-max kullan
normalized = np.zeros_like(Features, dtype=float)
for i in range(Features.shape[0]):  # 10 zaman
    for j in range(Features.shape[1]):  # 100 kişi
        normalized[i, j] = (Features[i, j] - data_min) / (data_max - data_min)

# 3. BENZERLİK SKORU HESAPLAMA FONKSİYONLARI

def SkorHesapla(v1, v2):
    """
    Öklid mesafesinden benzerlik skoru hesaplar
    Formül: score = 1 / (1 + OklidMesafesi)
    """
    dist = np.linalg.norm(v1 - v2) #Öklid mesafesini hesapla
    return 1.0 / (1.0 + dist)

# 4. GENUINE SKORLARININ HESAPLANMASI

genuine_scores = []

# Her kişi için
for person_id in range(100):
    # Aynı kişinin 10 farklı zaman dilimindeki örnekleri
    person_samples = normalized[:, person_id, :]
    
    # 10 örnek arasındaki tüm ikili kombinasyonları
    for i, j in combinations(range(10), 2):
        score = SkorHesapla(person_samples[i], person_samples[j])
        genuine_scores.append(score)

genuine_scores = np.array(genuine_scores)

# 5. IMPOSTER SKORLARININ HESAPLANMASI

imposter_scores = []

# Farklı kişiler arasındaki karşılaştırmalar
for i, j in combinations(range(100), 2):
    # Her iki kişinin tüm zaman kombinasyonları
    for t1 in range(10):
        for t2 in range(10):
            score = SkorHesapla(normalized[t1, i], normalized[t2, j])
            imposter_scores.append(score)

imposter_scores = np.array(imposter_scores)

# 6. SKOR DAĞILIMLARININ GÖSTERİLMESİ

plt.figure(figsize=(10, 6))


# Genuine skorlar
plt.hist(genuine_scores, bins=50, alpha=0.6, color='green', 
         label='Genuine Scores)', 
         density=True, edgecolor='black', linewidth=0.5)

# Imposter skorlar
plt.hist(imposter_scores, bins=50, alpha=0.6, color='red', 
         label='Imposter Scores)', 
         density=True, edgecolor='black', linewidth=0.5)

plt.xlabel('Benzerlik Skoru', fontsize=12)
plt.ylabel('Yoğunluk', fontsize=12)
plt.title('Genuine ve Imposter Skor Dağılımları', fontsize=14, fontweight='bold')
plt.legend(loc='upper center', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# 7. FAR (False Acceptance Rate) ve FRR (False Reject Rate) HESAPLAMA

# Eşik değerleri 0'dan 1'e kadar küçük adımlarla
thresholds = np.linspace(0, 1, 1001)

far_values = []  # False Acceptance Rate
frr_values = []  # False Reject Rate

for threshold in thresholds:
    far = np.sum(imposter_scores >= threshold) / len(imposter_scores)
    far_values.append(far)

    frr = np.sum(genuine_scores < threshold) / len(genuine_scores)
    frr_values.append(frr)

far_values = np.array(far_values)
frr_values = np.array(frr_values)

# 8. EER (Equal Error Rate) HESAPLAMA

# FAR ve FRR'ın kesiştiği noktayı bul (mutlak farkın minimum olduğu yer)
eer_idx = np.argmin(np.abs(far_values - frr_values))
eer_threshold = thresholds[eer_idx]
eer_value = (far_values[eer_idx] + frr_values[eer_idx]) / 2

print(f"EER Threshold: {eer_threshold:.4f}")
print(f"EER Değeri: {eer_value:.4f} (%{eer_value*100:.2f})")
print(f"FAR: {far_values[eer_idx]:.4f}")
print(f"FRR: {frr_values[eer_idx]:.4f}")

# 9. FAR-FRR-EER GRAFİĞİ (Eşik Değerine Göre)

plt.figure(figsize=(10, 6))

plt.plot(thresholds, far_values, 'r-', linewidth=2, label='FAR')
plt.plot(thresholds, frr_values, 'b-', linewidth=2, label='FRR')

# EER noktasını işaretle
plt.axvline(x=eer_threshold, color='green', linestyle='--', 
            label=f'EER Threshold = {eer_threshold:.3f}')
plt.scatter([eer_threshold], [eer_value], color='green', s=100, zorder=5)
plt.annotate(f'EER = {eer_value:.4f}', 
             xy=(eer_threshold, eer_value), 
             xytext=(eer_threshold + 0.1, eer_value + 0.1),
             fontsize=10, color='green',
             arrowprops=dict(arrowstyle='->', color='green'))

plt.xlabel('Eşik Değeri', fontsize=12)
plt.ylabel('Hata Oranı', fontsize=12)
plt.title('Eşik Değerine Göre FAR ve FRR Değişimi', 
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 10. FAR'a Karşı FRR
plt.figure(figsize=(10, 6))

# Sıfır değerlerden kaçınmak için küçük epsilon ekle
epsilon = 1e-10
far_log = np.log10(far_values + epsilon)
frr_log = np.log10(frr_values + epsilon)

plt.plot(far_values, frr_values, 'b-', linewidth=2, label='DET Curve')

# EER noktasını işaretle
plt.scatter([far_values[eer_idx]], [frr_values[eer_idx]], 
            color='red', s=100, zorder=5, label=f'EER = {eer_value:.4f}')

# EER çizgisi
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x (EER Hattı)')

plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
plt.ylabel('False Reject Rate (FRR)', fontsize=12)
plt.title('FAR\'a Karşı FRR', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
