import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

# Чтобы кириллица обычно отображалась корректно
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

os.makedirs("images", exist_ok=True)

# Дано: Exp(1), p(x)=e^{-x}, x>=0; n=25
rng = np.random.default_rng(42)
n = 25
x = rng.exponential(scale=1.0, size=n)
xs = np.sort(x)

print("Выборка (n=25):")
print(np.round(x, 6))
print("\nВариационный ряд:")
print(np.round(xs, 6))

# ============================================================
# (a) Мода, медиана, размах, коэффициент асимметрии
# ============================================================

# Число интервалов по формуле Стерджесса
k = int(np.ceil(1 + np.log2(n)))

# Оценка моды по KDE (и дополнительно по модальному интервалу гистограммы)
kde_x = gaussian_kde(x)
grid_x = np.linspace(0, xs[-1] * 1.1, 2000)
mode_kde = float(grid_x[np.argmax(kde_x(grid_x))])

counts, edges = np.histogram(x, bins=k)
imax = int(np.argmax(counts))
mode_hist = float(0.5 * (edges[imax] + edges[imax + 1]))

median = float(np.median(x))
data_range = float(np.max(x) - np.min(x))
skew = float(stats.skew(x, bias=False))

print("\n(a) Описательные характеристики:")
print(f"Мода (по KDE)                  ≈ {mode_kde:.6f}")
print(f"Мода (по модальному интервалу) ≈ {mode_hist:.6f}")
print(f"Медиана                        = {median:.6f}")
print(f"Размах                         = {data_range:.6f}")
print(f"Коэффициент асимметрии         = {skew:.6f}")

# ============================================================
# (b) Эмпирическая функция распределения, гистограмма, boxplot
# ============================================================

# Эмпирическая функция распределения F_n(x)
y = np.arange(1, n + 1) / n
plt.figure()
plt.step(xs, y, where="post")
plt.xlabel("x")
plt.ylabel("F_n(x)")
plt.title("Эмпирическая функция распределения")
plt.grid(True)
plt.savefig("images/ecdf.png", dpi=220, bbox_inches="tight")
plt.close()

# Гистограмма плотности + теоретическая плотность e^{-x}
plt.figure()
plt.hist(x, bins=k, density=True, alpha=0.6, edgecolor="black")
grid_pdf = np.linspace(0, max(6, xs[-1] * 1.1), 400)
plt.plot(grid_pdf, np.exp(-grid_pdf))
plt.xlabel("x")
plt.ylabel("Плотность")
plt.title("Гистограмма (плотность) и теоретическая плотность e^{-x}")
plt.grid(True)
plt.savefig("images/hist.png", dpi=220, bbox_inches="tight")
plt.close()

# Boxplot
plt.figure()
plt.boxplot(x, vert=False)
plt.xlabel("x")
plt.title("Boxplot")
plt.savefig("images/box.png", dpi=220, bbox_inches="tight")
plt.close()

# ============================================================
# Bootstrap-выборки (общие для (c)-(e))
# ============================================================

B = 20000
idx = rng.integers(0, n, size=(B, n))
boot = x[idx]

# ============================================================
# (c) Плотность распределения среднего: ЦПТ vs bootstrap
# ============================================================

boot_means = boot.mean(axis=1)

kde_mean = gaussian_kde(boot_means)
grid_m = np.linspace(min(boot_means.min(), 0.0), boot_means.max(), 400)
boot_mean_pdf = kde_mean(grid_m)

# ЦПТ (plug-in): N(x̄, s^2/n)
mu_hat = float(np.mean(x))
s_hat = float(np.std(x, ddof=1))
clt_pdf = stats.norm.pdf(grid_m, loc=mu_hat, scale=s_hat / np.sqrt(n))

plt.figure()
plt.plot(grid_m, boot_mean_pdf, label="bootstrap (KDE)")
plt.plot(grid_m, clt_pdf, label="ЦПТ: N(x̄, s²/n)")
plt.xlabel("Среднее")
plt.ylabel("Плотность")
plt.title("Плотность распределения среднего: ЦПТ и bootstrap")
plt.grid(True)
plt.legend()
plt.savefig("images/mean_clt_vs_boot.png", dpi=220, bbox_inches="tight")
plt.close()

# ============================================================
# (d) Bootstrap-плотность асимметрии и P(skew < 1)
# ============================================================

boot_skews = stats.skew(boot, axis=1, bias=False)
p_skew_lt_1 = float(np.mean(boot_skews < 1.0))

print(f"\n(d) Bootstrap-оценка P(коэффициент асимметрии < 1) ≈ {p_skew_lt_1:.4f}")

kde_sk = gaussian_kde(boot_skews)
grid_s = np.linspace(boot_skews.min(), boot_skews.max(), 400)

plt.figure()
plt.plot(grid_s, kde_sk(grid_s))
plt.axvline(1.0, linestyle="--")
plt.xlabel("Коэффициент асимметрии")
plt.ylabel("Плотность")
plt.title("Bootstrap-плотность коэффициента асимметрии (линия: 1)")
plt.grid(True)
plt.savefig("images/skew_boot.png", dpi=220, bbox_inches="tight")
plt.close()

# ============================================================
# (e) Плотность распределения медианы: асимптотика vs bootstrap
# ============================================================

boot_medians = np.median(boot, axis=1)
kde_med = gaussian_kde(boot_medians)
grid_md = np.linspace(boot_medians.min(), boot_medians.max(), 400)

# Для Exp(1): m = ln 2, f(m)=1/2 => Var(median) ~ 1/n
m_true = float(np.log(2))
med_asym_pdf = stats.norm.pdf(grid_md, loc=m_true, scale=1.0 / np.sqrt(n))

plt.figure()
plt.plot(grid_md, kde_med(grid_md), label="bootstrap (KDE)")
plt.plot(grid_md, med_asym_pdf, label="Асимпт.: N(ln 2, 1/n)")
plt.xlabel("Медиана")
plt.ylabel("Плотность")
plt.title("Плотность распределения медианы: асимптотика и bootstrap")
plt.grid(True)
plt.legend()
plt.savefig("images/median_asym_vs_boot.png", dpi=220, bbox_inches="tight")
plt.close()

print("\nСохранены графики (PNG) в папку images/:")
print("ecdf.png, hist.png, box.png, mean_clt_vs_boot.png, skew_boot.png, median_asym_vs_boot.png")