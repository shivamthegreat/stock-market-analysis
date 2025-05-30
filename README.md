# 📊 Stock Market Analysis Dashboard

Welcome to the **Stock Market Analysis Dashboard**! This project, built by Shivam, provides a sleek, interactive dashboard for visualizing and downloading processed stock data for major tech stocks (AAPL, MSFT, NFLX, GOOG).

---

## 🔍 Features

* **Dark/Light Mode Toggle**: Easily switch between themes for optimal viewing.
* **Interactive Charts**: Click on any chart thumbnail to view a full-screen lightbox.
* **Animations & Effects**:

  * Smooth hover and entrance animations on figures.
  * Subtle neon glow accents in dark mode.
  * Aesthetic leaf-fall animation for added flair.
* **Download Processed Data**: Quickly export the cleaned, processed CSV file with one click.

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/shivamthegreat/stock-market-analysis.git
   cd stock-market-analysis
   ```

2. **Assets**

   * Ensure your `images/` folder contains the eight key charts:

     * `price_trends.png`
     * `correlation_matrix.png`
     * `average_volume.png`
     * `returns_distribution.png`
     * `rolling_correlation.png`
     * `normalized_comparison.png`
     * `cumulative_returns.png`
     * `weekly_patterns.png`
   * Replace the header GIF with your own image by placing it in `images/your-new-image.png`.

3. **Open the Dashboard**

   * Simply open `index.html` in your browser to view the dashboard locally.

---

## 📁 Repository Structure

```
stock-market-analysis/
├── images/                 # Chart images and header banner
├── processed_stock_data.csv  # Exportable CSV of processed stock data
├── index.html              # Main HTML dashboard
└── README.md               # Project overview and instructions
```

---

## 🎨 Customization

* **Theme Colors**: Modify CSS variables in `<style>` for accent/neon hues.
* **Leaf Animation**: Adjust or replace emojis in the `.leaf` divs, or swap in custom SVGs via CSS.
* **Chart Animations**: Tweak keyframe durations and easing in CSS.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for:

* Additional interactivity (e.g., data filtering).
* Performance optimizations.
* Design enhancements.

Please ensure your code follows the existing style and includes descriptive commit messages.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

© 2025 Shivam
