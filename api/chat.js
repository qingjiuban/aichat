export default async function handler(req, res) {
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

    const { model, prompt, temperature = 0.7, topP = 0.9, freqPenalty = 0, maxTokens = 2000 } = req.body;
    if (!model || !prompt) return res.status(400).json({ error: 'Missing model or prompt' });

    try {
        let content;
        if (model.startsWith('zhipu:')) {
            const zhipuModel = model.slice(6);
            content = await callZhipu(zhipuModel, prompt, temperature, topP, freqPenalty, maxTokens);
        } else {
            throw new Error('Unsupported model provider. Only zhipu: is supported.');
        }
        res.status(200).json({ content });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
}

async function callZhipu(model, prompt, temperature, topP, freqPenalty, maxTokens) {
    const apiKey = process.env.ZHIPU_API_KEY;
    if (!apiKey) throw new Error('Missing Zhipu API Key. Please set ZHIPU_API_KEY in Vercel environment variables.');

    const response = await fetch('https://open.bigmodel.cn/api/paas/v4/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: maxTokens,
            temperature: temperature,
            top_p: topP,
            frequency_penalty: freqPenalty
        })
    });
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Zhipu API error (${response.status}): ${errorText}`);
    }
    const data = await response.json();
    return data.choices[0].message.content;
}