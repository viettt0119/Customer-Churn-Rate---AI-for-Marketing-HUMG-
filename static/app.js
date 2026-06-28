// app.js - Client side logic for ChurnShield Dashboard

// Predefined customer profiles for quick population
const PRESETS = {
    loyal: {
        CreditScore: 780,
        Age: 32,
        Geography: 'France',
        Gender: 'Female',
        Tenure: 8,
        Balance: 0.00,
        EstimatedSalary: 95000.00,
        NumOfProducts: 2,
        HasCrCard: true,
        IsActiveMember: true
    },
    high_risk: {
        CreditScore: 490,
        Age: 56,
        Geography: 'Germany',
        Gender: 'Female',
        Tenure: 1,
        Balance: 145000.00,
        EstimatedSalary: 120000.00,
        NumOfProducts: 3,
        HasCrCard: false,
        IsActiveMember: false
    },
    average: {
        CreditScore: 615,
        Age: 42,
        Geography: 'Spain',
        Gender: 'Male',
        Tenure: 4,
        Balance: 78000.00,
        EstimatedSalary: 62000.00,
        NumOfProducts: 1,
        HasCrCard: true,
        IsActiveMember: false
    }
};

// SVG circular progress max stroke-dashoffset (2 * PI * r = 2 * 3.14159 * 80 ≈ 502)
const MAX_DASH_OFFSET = 502;

document.addEventListener('DOMContentLoaded', () => {
    // Sliders
    const sliders = [
        { inputId: 'CreditScore', valId: 'credit-val' },
        { inputId: 'Age', valId: 'age-val' },
        { inputId: 'Tenure', valId: 'tenure-val' }
    ];

    sliders.forEach(slider => {
        const inputEl = document.getElementById(slider.inputId);
        const valEl = document.getElementById(slider.valId);
        
        inputEl.addEventListener('input', (e) => {
            valEl.textContent = e.target.value;
        });
    });

    // Load Model Info
    fetchModelMetadata();

    // Form submission
    const form = document.getElementById('churn-form');
    form.addEventListener('submit', handleFormSubmit);
});

// Load Model Health Info from /health
async function fetchModelMetadata() {
    const activeModelEl = document.getElementById('active-model');
    try {
        const response = await fetch('/health');
        if (!response.ok) throw new Error('Health check failed');
        
        const data = await response.json();
        if (data.status === 'Healthy') {
            const f1Metric = data.metrics?.f1_score_weighted;
            const f1Str = f1Metric ? `(F1: ${(f1Metric * 100).toFixed(1)}%)` : '';
            activeModelEl.textContent = `${data.model_name} ${f1Str}`;
            
            // Also store F1 globally to render in the results footer later
            window.activeModelF1 = f1Metric ? `${(f1Metric * 100).toFixed(2)}%` : 'N/A';
        } else {
            activeModelEl.textContent = 'Degraded (No Model)';
            activeModelEl.style.color = '#ee5253';
        }
    } catch (error) {
        console.error('Error fetching model metadata:', error);
        activeModelEl.textContent = 'Offline (Error)';
        activeModelEl.style.color = '#ee5253';
    }
}

// Function to load presets
function loadPreset(profileKey) {
    const data = PRESETS[profileKey];
    if (!data) return;

    // Apply values to inputs
    document.getElementById('CreditScore').value = data.CreditScore;
    document.getElementById('credit-val').textContent = data.CreditScore;

    document.getElementById('Age').value = data.Age;
    document.getElementById('age-val').textContent = data.Age;

    document.getElementById('Geography').value = data.Geography;
    document.getElementById('Gender').value = data.Gender;

    document.getElementById('Tenure').value = data.Tenure;
    document.getElementById('tenure-val').textContent = data.Tenure;

    document.getElementById('Balance').value = data.Balance.toFixed(2);
    document.getElementById('EstimatedSalary').value = data.EstimatedSalary.toFixed(2);

    // Set product radio buttons
    const radioBtn = document.getElementById(`prod-${data.NumOfProducts}`);
    if (radioBtn) radioBtn.checked = true;

    // Set toggles
    document.getElementById('HasCrCard').checked = data.HasCrCard;
    document.getElementById('IsActiveMember').checked = data.IsActiveMember;
    
    // Add micro-animation styling to presets buttons
    const btn = event.currentTarget;
    btn.classList.add('active');
    setTimeout(() => btn.classList.remove('active'), 300);
}

// Handle submit
async function handleFormSubmit(e) {
    e.preventDefault();

    const placeholder = document.getElementById('results-placeholder');
    const loader = document.getElementById('results-loader');
    const content = document.getElementById('results-content');
    const submitBtn = document.getElementById('submit-btn');

    // 1. Set loading UI state
    placeholder.classList.add('hidden');
    content.classList.add('hidden');
    loader.classList.remove('hidden');
    submitBtn.disabled = true;

    // 2. Build Request payload
    const form = e.target;
    const numProductsInput = form.querySelector('input[name="NumOfProducts"]:checked');
    
    const payload = {
        CreditScore: parseInt(document.getElementById('CreditScore').value, 10),
        Geography: document.getElementById('Geography').value,
        Gender: document.getElementById('Gender').value,
        Age: parseInt(document.getElementById('Age').value, 10),
        Tenure: parseInt(document.getElementById('Tenure').value, 10),
        Balance: parseFloat(document.getElementById('Balance').value),
        NumOfProducts: parseInt(numProductsInput ? numProductsInput.value : 1, 10),
        HasCrCard: document.getElementById('HasCrCard').checked ? 1 : 0,
        IsActiveMember: document.getElementById('IsActiveMember').checked ? 1 : 0,
        EstimatedSalary: parseFloat(document.getElementById('EstimatedSalary').value)
    };

    try {
        // Post prediction query
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Prediction failed');
        }

        const result = await response.json();
        
        // Render Output
        renderPrediction(result, payload);

    } catch (error) {
        console.error('Error during prediction:', error);
        alert(`Prediction Error: ${error.message}`);
        placeholder.classList.remove('hidden');
    } finally {
        loader.classList.add('hidden');
        submitBtn.disabled = false;
    }
}

// Render prediction outputs onto the UI
function renderPrediction(result, inputs) {
    const content = document.getElementById('results-content');
    const percentageText = document.getElementById('risk-percentage');
    const gaugeProgress = document.getElementById('gauge-progress');
    const riskBadge = document.getElementById('risk-badge');
    const riskIcon = document.getElementById('risk-icon');
    const riskStatus = document.getElementById('risk-status');
    const servingModel = document.getElementById('serving-model');
    const modelF1 = document.getElementById('model-f1');
    const insightsList = document.getElementById('insights-list');
    const recommendationText = document.getElementById('recommendation-text');

    const prob = result.probability; // probability is decimal (e.g. 0.8124)
    const probPercentage = Math.round(prob * 100);

    // 1. Update text & metadata
    percentageText.textContent = `${probPercentage}%`;
    servingModel.textContent = result.model_name;
    modelF1.textContent = window.activeModelF1 || 'N/A';

    // 2. Animate Circular Progress Gauge
    const offset = MAX_DASH_OFFSET - (MAX_DASH_OFFSET * prob);
    gaugeProgress.style.strokeDashoffset = offset;

    // Reset indicator classes
    riskBadge.className = 'risk-badge';
    gaugeProgress.classList.remove('state-safe', 'state-warning', 'state-danger');

    // 3. Determine Risk Classification Level
    let riskLevel = 'Low';
    let riskClass = 'state-safe';
    let iconClass = 'fa-circle-check';
    let recText = '';

    if (prob >= 0.60) {
        riskLevel = 'High Risk';
        riskClass = 'state-danger';
        iconClass = 'fa-triangle-exclamation';
    } else if (prob >= 0.30) {
        riskLevel = 'Medium Risk';
        riskClass = 'state-warning';
        iconClass = 'fa-circle-exclamation';
    }

    riskBadge.classList.add(riskClass);
    gaugeProgress.classList.add(riskClass);
    riskIcon.className = `fa-solid ${iconClass}`;
    riskStatus.textContent = riskLevel;

    // 4. Generate Dynamically Tailored Insights
    const insights = [];
    
    if (inputs.Age >= 50) {
        insights.push({
            icon: 'fa-user-clock',
            text: `High-risk age bracket detected: Customer is ${inputs.Age} years old. Older customer cohorts show higher historical churn.`
        });
    }
    if (inputs.IsActiveMember === 0) {
        insights.push({
            icon: 'fa-circle-nodes',
            text: 'Customer is inactive. Low engagement level is the strongest indicator of churn.'
        });
    }
    if (inputs.NumOfProducts >= 3) {
        insights.push({
            icon: 'fa-cubes',
            text: `Customer has ${inputs.NumOfProducts} bank products. High multi-product counts frequently correlate with product overload or dissatisfaction.`
        });
    }
    if (inputs.Geography === 'Germany') {
        insights.push({
            icon: 'fa-earth-europe',
            text: 'Geography alert: German branches historically exhibit higher churn rates compared to French and Spanish branches.'
        });
    }
    if (inputs.CreditScore < 500) {
        insights.push({
            icon: 'fa-credit-card',
            text: `Low Credit Score (${inputs.CreditScore}) may limit banking options or cause credit dissatisfaction.`
        });
    }
    if (inputs.Balance > 120000 && inputs.IsActiveMember === 0) {
        insights.push({
            icon: 'fa-piggy-bank',
            text: 'Significant idle wealth (High balance with low activity status). Subject to competitor takeover.'
        });
    }

    // Default insight if none of the specific alerts triggers
    if (insights.length === 0) {
        insights.push({
            icon: 'fa-shield-heart',
            text: 'Customer displays steady engagement profile, stable product configurations, and moderate balances.'
        });
    }

    // Render insights list HTML
    insightsList.innerHTML = insights.map(ins => `
        <li>
            <i class="fa-solid ${ins.icon}"></i>
            <span>${ins.text}</span>
        </li>
    `).join('');

    // 5. Generate Dynamically Tailored Recommendations
    if (prob >= 0.60) {
        recText = '<strong>High Priority Action Required:</strong> Assign a dedicated customer retention officer immediately. Arrange a courtesy call to investigate points of friction. Offer specialized incentives (e.g., fee waivers or premium account upgrades) and encourage them to enroll in automatic payments to secure engagement.';
    } else if (prob >= 0.30) {
        recText = '<strong>Proactive Relationship Building:</strong> Send a personalized product engagement survey. Recommend active products like credit cards or cross-sale incentives matching their estimated salary cohort. Incentivize mobile app log-ins with rewards programs to lift active membership status.';
    } else {
        recText = '<strong>Nurture & Retain:</strong> Keep normal marketing cadence. Highlight appreciation programs. Monitor product utilization and offer rewards for tenure markers (currently at ' + inputs.Tenure + ' years with the bank).';
    }
    recommendationText.innerHTML = recText;

    // 6. Make Content Visible
    content.classList.remove('hidden');
}
