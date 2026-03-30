export default async function handler(req, res) {
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

    const { model, prompt, temperature = 0.7, topP = 0.9, freqPenalty = 0, maxTokens = 1000 } = req.body;
    if (!model || !prompt) return res.status(400).json({ error: 'Missing model or prompt' });

    try {
        let content;
        if (model.startsWith('hf:')) {
            const hfModel = model.slice(3);
            content = await callHuggingFace(hfModel, prompt, maxTokens, temperature);
        } else if (model.startsWith('openrouter:')) {
            const orModel = model.slice(11);
            content = await callOpenRouter(orModel, prompt, maxTokens, temperature);
        } else if (model.startsWith('openai:')) {
            const oaiModel = model.slice(7);
            content = await callOpenAI(oaiModel, prompt, maxTokens, temperature, topP, freqPenalty);
        } else if (model.startsWith('google:')) {
            const googleModel = model.slice(7);
            content = await callGoogle(googleModel, prompt, maxTokens, temperature);
        } else if (model.startsWith('anthropic:')) {
            const antModel = model.slice(10);
            content = await callAnthropic(antModel, prompt, maxTokens, temperature);
        } else {
            throw new Error('Unsupported model provider');
        }
        res.status(200).json({ content });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
}

async function callHuggingFace(model, prompt, maxTokens, temperature) {
    const apiKey = process.env.HF_API_KEY;
    if (!apiKey) throw new Error('Missing Hugging Face API Key');
    const url = `https://api-inference.huggingface.co/models/${model}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: prompt, parameters: { max_new_tokens: maxTokens, temperature, return_full_text: false } })
    });
    if (!response.ok) throw new Error(`HF API error: ${response.status}`);
    const data = await response.json();
    return data[0]?.generated_text || data.generated_text;
}

async function callOpenRouter(model, prompt, maxTokens, temperature) {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) throw new Error('Missing OpenRouter API Key');
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens, temperature })
    });
    if (!response.ok) throw new Error(`OpenRouter error: ${response.status}`);
    const data = await response.json();
    return data.choices[0].message.content;
}

async function callOpenAI(model, prompt, maxTokens, temperature, topP, freqPenalty) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('Missing OpenAI API Key');
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens, temperature, top_p: topP, frequency_penalty: freqPenalty })
    });
    if (!response.ok) throw new Error(`OpenAI API error: ${response.status}`);
    const data = await response.json();
    return data.choices[0].message.content;
}

async function callGoogle(model, prompt, maxTokens, temperature) {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) throw new Error('Missing Google API Key');
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { maxOutputTokens: maxTokens, temperature } })
    });
    if (!response.ok) throw new Error(`Google API error: ${response.status}`);
    const data = await response.json();
    return data.candidates[0].content.parts[0].text;
}

async function callAnthropic(model, prompt, maxTokens, temperature) {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('Missing Anthropic API Key');
    const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'x-api-key': apiKey, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens, temperature })
    });
    if (!response.ok) throw new Error(`Anthropic API error: ${response.status}`);
    const data = await response.json();
    return data.content[0].text;
}
