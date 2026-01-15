"""
HTML templates for training reports

Contains the base HTML template structure.
Separated from generator logic for easier maintenance.
"""


def get_html_template() -> str:
    """
    Get the base HTML template for training reports.
    
    The template contains placeholders that will be replaced by the generator:
    - {{MODEL_NAME}}, {{DATE}}, {{PLUGIN_VERSION}}
    - {{ARCHITECTURE}}, {{DATASET_SIZE}}, {{EPOCHS}}, etc.
    - {{DATASET_TABLE}}, {{METRICS_TABLE}}, {{RECOMMENDATIONS}}
    - {{LR_FINDER_SECTION}}, {{QUALITATIVE_EXAMPLES_SECTION}}, {{CONFUSION_MATRIX_SECTION}}
    
    Returns:
        HTML template as string
    """
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Training Report — QModel Trainer</title>
    <style>
      :root {
        --bg: #fbfcfe;
        --card: #ffffff;
        --muted: #6b7280;
        --accent: #0b63c6;
        --accent-2: #0b9bd6;
        --good: #16a34a;
        --bad: #ef4444;
        --mono: Menlo, monospace;
      }
      html, body {
        height: 100%;
        margin: 0;
        font-family: Inter, Segoe UI, Arial, sans-serif;
        background: var(--bg);
        color: #111827;
      }
      .container {
        max-width: 1100px;
        margin: 28px auto;
        padding: 24px;
      }
      header {
        display: flex;
        align-items: center;
        gap: 16px;
      }
      .brand {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 12px;
      }
      .logo {
        width: 64px;
        height: 64px;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
      }
      h1 {
        margin: 0;
        font-size: 20px;
      }
      .meta {
        color: var(--muted);
        font-size: 13px;
      }
      .grid {
        display: grid;
        grid-template-columns: 1fr 360px;
        gap: 20px;
        margin-top: 18px;
      }
      .card {
        background: var(--card);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
      }
      .section-title {
        font-weight: 700;
        margin-bottom: 8px;
      }
      .kvs {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }
      .kv {
        background: #f8fafc;
        padding: 10px;
        border-radius: 8px;
        min-width: 140px;
      }
      .kv b {
        display: block;
        font-size: 14px;
      }
      .small {
        font-size: 13px;
        color: var(--muted);
      }
      .chart {
        border-radius: 8px;
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--muted);
        overflow: hidden;
      }
      .chart img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        cursor: pointer;
        transition: transform 0.2s ease;
      }
      .chart img:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      }
      
      /* Lightbox styles */
      .lightbox {
        display: none;
        position: fixed;
        z-index: 9999;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.9);
        animation: fadeIn 0.2s ease;
      }
      .lightbox.active {
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .lightbox-content {
        max-width: 95%;
        max-height: 95%;
        object-fit: contain;
        animation: zoomIn 0.3s ease;
      }
      .lightbox-close {
        position: absolute;
        top: 20px;
        right: 35px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.2s;
      }
      .lightbox-close:hover {
        color: #bbb;
      }
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
      @keyframes zoomIn {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
      }
      .click-hint {
        font-size: 12px;
        color: #6b7280;
        text-align: center;
        margin-top: 4px;
        font-style: italic;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      th, td {
        padding: 8px;
        border-bottom: 1px solid #f1f5f9;
        text-align: left;
      }
      .footer {
        margin-top: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: var(--muted);
        font-size: 13px;
      }
      .explain {
        font-size: 13px;
        color: var(--muted);
        margin-top: 8px;
      }
      .two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .notice {
        padding: 10px;
        border-radius: 8px;
        background: #fff7ed;
        color: #92400e;
        font-size: 13px;
        margin-top: 8px;
      }
      a.link {
        color: var(--accent);
        text-decoration: none;
      }
      .disclaimer {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 16px;
      }
      /* Qualitative Examples Styles */
      .iou-dropdown {
        width: 100%;
        padding: 10px 14px;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        background: white;
        font-size: 14px;
        font-weight: 500;
        color: #374151;
        cursor: pointer;
        transition: all 0.2s ease;
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg width='12' height='8' viewBox='0 0 12 8' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1.5L6 6.5L11 1.5' stroke='%23374151' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 14px center;
        padding-right: 40px;
      }
      .iou-dropdown:hover {
        border-color: var(--accent);
        background-color: #f9fafb;
      }
      .iou-dropdown:focus {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(11, 99, 198, 0.1);
      }
      .examples-grid {
        display: flex;
        flex-direction: column;
        gap: 20px;
      }
      .example-triplet {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background: #f9fafb;
      }
      .triplet-header {
        font-weight: 700;
        margin-bottom: 12px;
        color: #374151;
      }
      .triplet-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
      }
      .triplet-col {
        text-align: center;
      }
      .triplet-label {
        font-size: 12px;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .triplet-col img {
        width: 100%;
        border-radius: 6px;
        border: 1px solid #d1d5db;
        cursor: pointer;
        transition: transform 0.2s ease;
      }
      .triplet-col img:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      /* YOLO validation pairs */
      .example-pair {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background: #f9fafb;
        margin-bottom: 16px;
      }
      .pair-header {
        font-weight: 700;
        margin-bottom: 12px;
        color: #374151;
      }
      .pair-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
      }
      .pair-col {
        text-align: center;
      }
      .pair-label {
        font-size: 12px;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .pair-col img {
        width: 100%;
        border-radius: 6px;
        border: 1px solid #d1d5db;
        cursor: pointer;
        transition: transform 0.2s ease;
      }
      .pair-col img:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <div class="brand">
          <div class="logo">QT</div>
          <div>
            <h1>{{MODEL_NAME}} — Training Report</h1>
            <div class="meta">
              Model trained on <strong>{{DATE}}</strong> • QModel Trainer v{{PLUGIN_VERSION}}
            </div>
          </div>
        </div>
      </header>

      <div class="grid">
        <div>
          <!-- Executive Summary -->
          <div class="card">
            <div class="section-title">Summary</div>
            <div class="kvs">
              <div class="kv">
                <b>{{ARCHITECTURE}}</b><span class="small">Model Type</span>
              </div>
              <div class="kv">
                <b>{{DATASET_SIZE}}</b><span class="small">Training Images</span>
              </div>
              <div class="kv">
                <b>{{EPOCHS}}</b><span class="small">Epochs</span>
              </div>
              <div class="kv">
                <b>{{LEARNING_RATE}}</b><span class="small">Learning Rate</span>
              </div>
              <div class="kv">
                <b>{{BEST_METRIC}}</b><span class="small">{{BEST_METRIC_LABEL}}</span>
              </div>
              <div class="kv">
                <b>{{TRAIN_TIME}}</b><span class="small">Total Duration</span>
              </div>
            </div>
            <div style="margin-top: 14px" class="explain">
              <strong>Summary:</strong> {{SUMMARY_TEXT}}
            </div>
          </div>

          <!-- Dataset details -->
          <div class="card" style="margin-top: 16px">
            <div class="section-title">Dataset Details</div>
            <div class="small">Classes and distribution</div>
            <div style="margin-top: 10px">
              {{DATASET_TABLE}}
            </div>
            <div class="explain">
              Tile size: <strong>{{TILE_SIZE}}</strong> • 
              Train/Val split: <strong>{{VAL_SPLIT}}</strong> •
              Batch size: <strong>{{BATCH_SIZE}}</strong>
            </div>
          </div>

          <!-- LR finder (if used) -->
          {{LR_FINDER_SECTION}}

          <!-- Training curves -->
          {{TRAINING_CURVES_SECTION}}

          <!-- Qualitative Examples -->
          {{QUALITATIVE_EXAMPLES_SECTION}}

          <!-- Confusion Matrix -->
          {{CONFUSION_MATRIX_SECTION}}

          <!-- Final Metrics -->
          <div class="card" style="margin-top: 16px">
            <div class="section-title">Validation Metrics (Best Model)</div>
            <div class="explain" style="margin-bottom:10px;">
              Metrics of the saved model (epoch {{BEST_EPOCH}}, highest IoU).
            </div>
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {{METRICS_TABLE}}
              </tbody>
            </table>
          </div>
        </div>

        <aside>
          <div class="card">
            <div class="section-title">Technical Information</div>
            <div class="small">Hardware & Environment</div>
            <div style="margin-top: 10px" class="kvs">
              {{TECHNICAL_INFO}}
            </div>
          </div>

          <div class="card" style="margin-top: 12px">
            <div class="section-title">Training Configuration</div>
            <div class="small">Hyperparameters</div>
            <div style="margin-top: 8px">
              <table>
                <tbody>
                  <tr><td><strong>Optimizer</strong></td><td>{{OPTIMIZER}}</td></tr>
                  <tr><td><strong>Learning Rate</strong></td><td>{{LEARNING_RATE}}</td></tr>
                  <tr><td><strong>Scheduler</strong></td><td>{{SCHEDULER}}</td></tr>
                  <tr><td><strong>Early Stopping</strong></td><td>{{EARLY_STOPPING}}</td></tr>
                  <tr><td><strong>Pretrained</strong></td><td>{{PRETRAINED}}</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="card" style="margin-top: 12px">
            <div class="section-title">Support & Resources</div>
            <div class="small">Need help improving your model?</div>
            <div style="margin-top: 10px">
              <div><a class="link" href="https://qgeoai.nextelia.fr/">QGeoAI Documentation</a></div>
            </div>
            <div class="disclaimer">
              Results are indicative and not guaranteed. • Professional support for validated workflows is available via <a class="link" href="https://www.nextelia.fr/">Nextelia</a>
            </div>
          </div>
        </aside>
      </div>

      <div class="footer">
        <div class="small">
          Report generated automatically on {{DATE}}
        </div>
        <div>
          <a class="link" href="https://www.nextelia.fr/">Contact</a>
        </div>
      </div>
    </div>
    
    <!-- Lightbox for image zoom -->
    <div id="lightbox" class="lightbox" onclick="closeLightbox()">
      <span class="lightbox-close">&times;</span>
      <img class="lightbox-content" id="lightbox-img">
    </div>
    
    <script>
      function openLightbox(img) {
        const lightbox = document.getElementById('lightbox');
        const lightboxImg = document.getElementById('lightbox-img');
        lightbox.classList.add('active');
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt;
      }
      
      function closeLightbox() {
        const lightbox = document.getElementById('lightbox');
        lightbox.classList.remove('active');
      }
      
      // Close lightbox with Escape key
      document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
          closeLightbox();
        }
      });
      // Qualitative examples switcher
      function switchExamples(category) {
        // Hide all example grids
        document.querySelectorAll('.examples-grid').forEach(grid => {
          grid.style.display = 'none';
        });
        
        // Show selected category
        const selectedGrid = document.querySelector(`.examples-grid[data-category="${category}"]`);
        if (selectedGrid) {
          selectedGrid.style.display = 'flex';
        }
      }
      
      // Initialize with medium IoU on page load
      document.addEventListener('DOMContentLoaded', function() {
        switchExamples('medium');
      });
    </script>
  </body>
</html>"""