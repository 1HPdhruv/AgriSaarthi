const API_BASE = 'http://localhost:8000/api'; // Update to Render URL in production

// Tab Logic
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    tabBtns.forEach(b => b.classList.remove('active'));
    tabPanes.forEach(p => p.classList.remove('active'));
    
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');

    if (btn.dataset.tab === 'history-tab') {
      loadHistory();
    }
  });
});

// Update pH Value Display
const phInput = document.getElementById('soil-ph');
const phVal = document.getElementById('ph-val');
if(phInput) {
    phInput.addEventListener('input', (e) => {
        phVal.textContent = e.target.value;
    });
}

// Image Preview
const leafImg = document.getElementById('leaf-img');
const imagePreview = document.getElementById('image-preview');
if(leafImg) {
    leafImg.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        }
    });
}

// Crop Recommendation Submit
const cropForm = document.getElementById('crop-form');
if (cropForm) {
  cropForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = cropForm.querySelector('button');
    btn.textContent = 'Loading...';
    btn.disabled = true;

    const state = document.getElementById('state-select').value;
    const lang = document.getElementById('lang-select').value;
    const soilPh = parseFloat(document.getElementById('soil-ph').value);
    const water = document.getElementById('water-select').value;

    // Simple state to zone mapping logic (you can expand this)
    const stateToZone = {
        'Kerala': 'Tropical', 'Punjab': 'Temperate', 'Rajasthan': 'Dry'
    };
    const climateZone = stateToZone[state] || 'Tropical';

    try {
      const res = await fetch(`${API_BASE}/recommend_crop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state, climate_zone: climateZone, soil_ph: soilPh, water, lang })
      });
      const data = await res.json();
      
      const resultsDiv = document.getElementById('crop-results');
      resultsDiv.innerHTML = '';
      resultsDiv.classList.remove('hidden');

      if (data.success && data.crops.length > 0) {
        data.crops.forEach(c => {
          resultsDiv.innerHTML += `
            <div class="crop-card">
              <h3>${c.crop}</h3>
              <p>💰 Profit Index: ${c.profit_index.toFixed(2)}</p>
              <p>💧 Water Need: ${c.water_need}</p>
              <p>🗓️ Sowing: ${c.sowing_months}</p>
              <p>🧪 Fertilizer: ${c.fertilizer}</p>
            </div>
          `;
        });
      } else {
        resultsDiv.innerHTML = `<p style="color: #ef4444;">⚠️ No matching crops found.</p>`;
      }
    } catch (err) {
        console.error(err);
        alert('Failed to fetch recommendations.');
    } finally {
        btn.textContent = 'Recommend Crop';
        btn.disabled = false;
    }
  });
}

// Disease Detection Submit
const diseaseForm = document.getElementById('disease-form');
if (diseaseForm) {
  diseaseForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = diseaseForm.querySelector('button');
    btn.textContent = 'Analyzing...';
    btn.disabled = true;

    const file = document.getElementById('leaf-img').files[0];
    if (!file) {
      alert("Please upload an image.");
      btn.textContent = 'Analyze Image';
      btn.disabled = false;
      return;
    }

    const formData = new FormData();
    formData.append('farmer_name', document.getElementById('farmer-name').value);
    formData.append('lang', document.getElementById('disease-lang').value);
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/detect_disease`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      const resultsDiv = document.getElementById('disease-results');
      resultsDiv.innerHTML = '';
      resultsDiv.classList.remove('hidden');

      if (data.success) {
        resultsDiv.innerHTML = `
          <h3 style="color:var(--primary); margin-bottom: 0.5rem;">Disease: ${data.disease}</h3>
          <p>🌾 <strong>Crop:</strong> ${data.crop}</p>
          <p style="margin-top:0.5rem;">💊 <strong>Remedy:</strong> ${data.remedy}</p>
          <p style="margin-top:0.5rem;">⚠️ <strong>Precautions:</strong> ${data.precautions}</p>
          <p style="margin-top:0.5rem; color:var(--text-muted);">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
        `;
      } else {
        resultsDiv.innerHTML = `<p style="color: #ef4444;">Analysis failed.</p>`;
      }
    } catch (err) {
        console.error(err);
        alert('Failed to analyze image.');
    } finally {
        btn.textContent = 'Analyze Image';
        btn.disabled = false;
    }
  });
}

// History Loading
async function loadHistory() {
  const tbody = document.getElementById('history-tbody');
  tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;">Loading...</td></tr>`;
  
  try {
    const res = await fetch(`${API_BASE}/history`);
    const data = await res.json();
    
    if (data.success && data.history.length > 0) {
      tbody.innerHTML = data.history.map(row => `
        <tr>
          <td>${row.timestamp ? new Date(row.timestamp).toLocaleDateString() : '-'}</td>
          <td>${row.farmer_name || '-'}</td>
          <td>${row.crop || '-'}</td>
          <td>${row.disease || '-'}</td>
          <td>${row.remedy_en || '-'}</td>
        </tr>
      `).join('');
    } else {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;">No history found.</td></tr>`;
    }
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#ef4444;">Failed to load history.</td></tr>`;
  }
}

document.getElementById('refresh-history')?.addEventListener('click', loadHistory);
