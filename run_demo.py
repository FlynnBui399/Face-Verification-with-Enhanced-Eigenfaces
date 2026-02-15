"""
Demo Face Recognition — Chạy sau khi đã train bằng run_experiment.py
Usage: python run_demo.py
"""
import os, sys
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.eigenfaces import BaselineEigenfaces, EnhancedEigenfaces
from src.linalg_scratch import PCA_Scratch
from src.metrics import cosine_distance_batch, euclidean_distance_batch


# ============================================================
# 1. LOAD MODELS
# ============================================================

def load_models():
    model_path = os.path.join(RESULTS_DIR, 'trained_models.npz')
    if not os.path.exists(model_path):
        print(f'ERROR: Chưa có model! Chạy "python run_experiment.py" trước.')
        sys.exit(1)

    print(f'Loading models from {model_path}...')
    data = np.load(model_path, allow_pickle=True)

    img_size = int(data['image_size'])
    n_comp = int(data['n_components'])

    # Baseline
    baseline = BaselineEigenfaces(n_components=n_comp, image_size=img_size)
    baseline.pca.mean_ = data['bl_mean']
    baseline.pca.components_ = data['bl_components']
    baseline.pca.eigenvalues_ = data['bl_eigenvalues']
    baseline.pca.explained_variance_ratio_ = data['bl_evr']
    baseline.pca.n_features_ = int(data['bl_n_features'])

    # Enhanced (HistEq)
    enhanced = EnhancedEigenfaces(
        scales=[img_size], n_components_per_scale=[n_comp],
        use_illumination_norm=True, use_ensemble=False,
        use_multiscale=False, illumination_method='hist_eq')
    fe = enhanced.feature_extractor
    fe.pca.mean_ = data['eh_mean']
    fe.pca.components_ = data['eh_components']
    fe.pca.eigenvalues_ = data['eh_eigenvalues']
    fe.pca.explained_variance_ratio_ = data['eh_evr']
    fe.pca.n_features_ = int(data['eh_n_features'])

    # Results (cho threshold)
    baseline_results = data['baseline_results'].item()
    enhanced_results = data['enhanced_results'].item()

    print(f'Loaded! (image: {img_size}x{img_size}, components: {n_comp})')
    return baseline, enhanced, img_size, n_comp, baseline_results, enhanced_results


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_face(filepath, size):
    """Load ảnh → grayscale → resize."""
    img = PILImage.open(filepath).convert('L').resize((size, size), PILImage.BILINEAR)
    return np.array(img, dtype=np.float32)


def cos_dist(f1, f2):
    return float(cosine_distance_batch(f1[np.newaxis], f2[np.newaxis])[0])


# ============================================================
# 3. GALLERY
# ============================================================

gallery = {}  # {name: [(image, feat_bl, feat_enh), ...]}


def register_person(name, image_paths, baseline, enhanced, size):
    """Đăng ký 1 người với nhiều ảnh."""
    if name not in gallery:
        gallery[name] = []
    for p in image_paths:
        img = load_face(p, size)
        fb = baseline.extract_features(img)
        fe = enhanced.extract_features(img)
        gallery[name].append((img, fb, fe))
    print(f'  Registered "{name}": {len(gallery[name])} photo(s)')


def identify(test_img, enhanced):
    """Nhận diện 1:N."""
    fe = enhanced.extract_features(test_img)
    results = []
    for name, entries in gallery.items():
        dists = [cos_dist(fe, e[2]) for e in entries]
        best_i = dists.index(min(dists))
        results.append({'name': name, 'distance': min(dists), 'photo': entries[best_i][0]})
    results.sort(key=lambda x: x['distance'])
    return results


def verify(img1, img2, baseline, enhanced):
    """Xác minh 1:1."""
    fb1, fe1 = baseline.extract_features(img1), enhanced.extract_features(img1)
    fb2, fe2 = baseline.extract_features(img2), enhanced.extract_features(img2)
    return {
        'cos_enhanced': cos_dist(fe1, fe2),
        'cos_baseline': cos_dist(fb1, fb2),
    }


# ============================================================
# 4. VISUALIZATION
# ============================================================

def show_identify_result(test_img, results, n_comp, baseline, size):
    recon = baseline.reconstruct(test_img, n_components=n_comp)
    n = min(len(results), 5)
    fig, axes = plt.subplots(1, n + 2, figsize=(4*(n+2), 4))

    axes[0].imshow(test_img, cmap='gray'); axes[0].set_title('Test Image', fontweight='bold')
    axes[1].imshow(recon, cmap='gray'); axes[1].set_title('Reconstruction')

    for i, r in enumerate(results[:n]):
        ax = axes[i+2]
        ax.imshow(r['photo'], cmap='gray')
        conf = max(0, (1.5 - r['distance']) / 1.5) * 100
        c = 'green' if i==0 and r['distance']<1.2 else 'orange' if r['distance']<1.4 else 'red'
        ax.set_title(f'#{i+1} {r["name"]}\nd={r["distance"]:.3f}\n{conf:.0f}%',
                     color=c, fontweight='bold' if i==0 else 'normal')
        if i == 0:
            for s in ax.spines.values(): s.set_edgecolor(c); s.set_linewidth(3)

    for ax in axes: ax.axis('off')
    best = results[0]
    verdict = best['name'] if best['distance'] < 1.2 else '???'
    c = 'green' if best['distance'] < 1.2 else 'red'
    plt.suptitle(f'Result: {verdict} (d={best["distance"]:.3f})', fontsize=14, fontweight='bold', color=c)
    plt.tight_layout(); plt.show()


def show_verify_result(img1, img2, scores, threshold, n_comp, baseline):
    is_same = scores['cos_enhanced'] < threshold
    r1 = baseline.reconstruct(img1, n_components=n_comp)
    r2 = baseline.reconstruct(img2, n_components=n_comp)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(img1, cmap='gray'); axes[0,0].set_title('Image 1')
    axes[0,1].imshow(r1, cmap='gray');   axes[0,1].set_title('Reconstruction')
    axes[0,2].imshow(np.abs(img1-r1), cmap='hot'); axes[0,2].set_title('|Residual|')
    axes[1,0].imshow(img2, cmap='gray'); axes[1,0].set_title('Image 2')
    axes[1,1].imshow(r2, cmap='gray');   axes[1,1].set_title('Reconstruction')

    axes[1,2].axis('off')
    verdict = 'SAME PERSON' if is_same else 'DIFFERENT'
    box_c = 'lightgreen' if is_same else '#ffcccc'
    txt = (f"Enhanced cos: {scores['cos_enhanced']:.4f}\n"
           f"Baseline cos: {scores['cos_baseline']:.4f}\n"
           f"Threshold:    {threshold:.4f}\n\n"
           f"VERDICT: {verdict}")
    axes[1,2].text(0.05, 0.95, txt, transform=axes[1,2].transAxes,
        fontsize=12, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=box_c, alpha=0.9))

    for ax in axes.flat: ax.axis('off')
    c = 'green' if is_same else 'red'
    plt.suptitle(f'Verification: {verdict}', fontsize=16, fontweight='bold', color=c)
    plt.tight_layout(); plt.show()


# ============================================================
# 5. INTERACTIVE MENU
# ============================================================

def main():
    baseline, enhanced, img_size, n_comp, bl_results, enh_results = load_models()
    threshold = enh_results.get('LFW', {}).get('best_threshold', 1.0)

    while True:
        print(f'\n{"="*50}')
        print(f'  FACE RECOGNITION DEMO')
        print(f'  Gallery: {len(gallery)} người')
        print(f'{"="*50}')
        print('  1. Đăng ký khuôn mặt')
        print('  2. Nhận diện (1:N)')
        print('  3. Xác minh (1:1)')
        print('  4. Xem gallery')
        print('  0. Thoát')

        choice = input('\nChọn: ').strip()

        if choice == '1':
            name = input('Tên người: ').strip()
            if not name:
                continue
            print(f'Nhập đường dẫn ảnh (cách nhau bởi dấu phẩy):')
            print(f'VD: photos/minh1.jpg, photos/minh2.jpg')
            paths_str = input('Paths: ').strip()
            paths = [p.strip() for p in paths_str.split(',') if p.strip()]
            valid = [p for p in paths if os.path.exists(p)]
            if not valid:
                print('Không tìm thấy file nào!')
                continue
            if len(valid) < len(paths):
                missing = [p for p in paths if p not in valid]
                print(f'Bỏ qua file không tồn tại: {missing}')
            register_person(name, valid, baseline, enhanced, img_size)

        elif choice == '2':
            if not gallery:
                print('Gallery trống! Đăng ký trước.')
                continue
            path = input('Đường dẫn ảnh test: ').strip()
            if not os.path.exists(path):
                print(f'Không tìm thấy: {path}')
                continue
            test_img = load_face(path, img_size)
            results = identify(test_img, enhanced)
            print(f'\nRanking:')
            for i, r in enumerate(results):
                print(f'  #{i+1} {r["name"]:<12} distance={r["distance"]:.4f}{"  <<< BEST" if i==0 else ""}')
            show_identify_result(test_img, results, n_comp, baseline, img_size)

        elif choice == '3':
            p1 = input('Đường dẫn ảnh 1: ').strip()
            p2 = input('Đường dẫn ảnh 2: ').strip()
            if not os.path.exists(p1) or not os.path.exists(p2):
                print('File không tồn tại!')
                continue
            img1, img2 = load_face(p1, img_size), load_face(p2, img_size)
            scores = verify(img1, img2, baseline, enhanced)
            is_same = scores['cos_enhanced'] < threshold
            print(f'\nCosine distance: {scores["cos_enhanced"]:.4f} (threshold: {threshold:.4f})')
            print(f'Verdict: {"SAME PERSON" if is_same else "DIFFERENT PERSON"}')
            show_verify_result(img1, img2, scores, threshold, n_comp, baseline)

        elif choice == '4':
            if not gallery:
                print('Gallery trống!')
            else:
                for name, entries in gallery.items():
                    print(f'  {name}: {len(entries)} ảnh')

        elif choice == '0':
            print('Bye!')
            break

        else:
            print('Chọn 0-4!')


if __name__ == '__main__':
    main()