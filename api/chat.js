export default async function handler(req, res) {
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

    const { model, prompt } = req.body;
    if (!model || !prompt) return res.status(400).json({ error: 'Missing model or prompt' });

    try {
        let content;
        if (model.startsWith('openai:')) {
            const openaiModel = model.slice(7);
            content = await callOpenAI(openaiModel, prompt);
        } else if (model.startsWith('google:')) {
            const googleModel = model.slice(7);
            content = await callGoogle(googleModel, prompt);
        } else if (model.startsWith('anthropic:')) {
            const anthropicModel = model.slice(10);
            content = await callAnthropic(anthropicModel, prompt);
        } else if (model.startsWith('moonshot:')) {
            const moonshotModel = model.slice(9);
            content = await callMoonshot(moonshotModel, prompt);
        } else if (model.startsWith('zhipu:')) {
            const zhipuModel = model.slice(6);
            content = await callZhipu(zhipuModel, prompt);
        } else if (model.startsWith('qwen:')) {
            const qwenModel = model.slice(5);
            content = await callQwen(qwenModel, prompt);
        } else if (model.startsWith('baidu:')) {
            const baiduModel = model.slice(6);
            content = await callBaidu(baiduModel, prompt);
        } else if (model.startsWith('spark:')) {
            const sparkModel = model.slice(6);
            content = await callSpark(sparkModel, prompt);
        } else if (model.startsWith('doubao:')) {
            const doubaoModel = model.slice(7);
            content = await callDoubao(doubaoModel, prompt);
        } else if (model.startsWith('hunyuan:')) {
            const hunyuanModel = model.slice(8);
            content = await callHunyuan(hunyuanModel, prompt);
        } else {
            throw new Error('Unsupported model provider');
        }
        res.status(200).json({ content });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
}

// ======================== OpenAI ========================
async function callOpenAI(model, prompt) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('Missing OpenAI API Key');
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 2000,
            temperature: 0.7
        })
    });
    if (!response.ok) throw new Error(`OpenAI error: ${response.status}`);
    const data = await response.json();
    return data.choices[0].message.content;
}

// ======================== Google Gemini ========================
async function callGoogle(model, prompt) {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) throw new Error('Missing Google API Key');
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { maxOutputTokens: 2000, temperature: 0.7 }
        })
    });
    if (!response.ok) throw new Error(`Google error: ${response.status}`);
    const data = await response.json();
    return data.candidates[0].content.parts[0].text;
}

// ======================== Anthropic Claude ========================
async function callAnthropic(model, prompt) {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('Missing Anthropic API Key');
    const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 2000,
            temperature: 0.7
        })
    });
    if (!response.ok) throw new Error(`Anthropic error: ${response.status}`);
    const data = await response.json();
    return data.content[0].text;
}

// ======================== 月之暗面 Kimi ========================
async function callMoonshot(model, prompt) {
    const apiKey = process.env.MOONSHOT_API_KEY;
    if (!apiKey) throw new Error('Missing Moonshot API Key');
    const response = await fetch('https://api.moonshot.cn/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 2000,
            temperature: 0.7
        })
    });
    if (!response.ok) throw new Error(`Moonshot error: ${response.status}`);
    const data = await response.json();
    return data.choices[0].message.content;
}

// ======================== 智谱 GLM ========================
async function callZhipu(model, prompt) {
    const apiKey = process.env.ZHIPU_API_KEY;
    if (!apiKey) throw new Error('Missing Zhipu API Key');
    const response = await fetch('https://open.bigmodel.cn/api/paas/v4/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 2000,
            temperature: 0.7
        })
    });
    if (!response.ok) throw new Error(`Zhipu error: ${response.status}`);
    const data = await response.json();
    return data.choices[0].message.content;
}

// ======================== 阿里云通义千问 ========================
async function callQwen(model, prompt) {
    const apiKey = process.env.QWEN_API_KEY;
    if (!apiKey) throw new Error('Missing Qwen API Key');
    const response = await fetch('https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: model,
            input: { messages: [{ role: 'user', content: prompt }] },
            parameters: { max_tokens: 2000, temperature: 0.7 }
        })
    });
    if (!response.ok) throw new Error(`Qwen error: ${response.status}`);
    const data = await response.json();
    return data.output.choices[0].message.content;
}

// ======================== 百度文心一言 ========================
async function callBaidu(model, prompt) {
    const apiKey = process.env.BAIDU_API_KEY;
    const secretKey = process.env.BAIDU_SECRET_KEY;
    if (!apiKey || !secretKey) throw new Error('Missing Baidu API credentials');
    // 获取 access_token
    const tokenRes = await fetch(`https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=${apiKey}&client_secret=${secretKey}`);
    const tokenData = await tokenRes.json();
    const accessToken = tokenData.access_token;
    const response = await fetch(`https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/${model}?access_token=${accessToken}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: 'user', content: prompt }] })
    });
    if (!response.ok) throw new Error(`Baidu error: ${response.status}`);
    const data = await response.json();
    return data.result;
}

// ======================== 科大讯飞星火 ========================
async function callSpark(model, prompt) {
    const apiKey = process.env.SPARK_API_KEY;
    const apiSecret = process.env.SPARK_API_SECRET;
    const appId = process.env.SPARK_APP_ID;
    if (!apiKey || !apiSecret || !appId) throw new Error('Missing Spark credentials');
    // 此处需要实现 WebSocket 调用，为了简化，使用 HTTP 代理（需官方支持 HTTP）
    // 星火官方推荐使用 WebSocket，我们使用通用 HTTP 方式（需用户配置）
    // 实际上星火 HTTP API 有独立 endpoint，这里仅作示例
    const url = `https://spark-api.cn-huabei-1.xf-yun.com/v2.1/chat`;
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            header: { app_id: appId },
            parameter: { chat: { domain: model, max_tokens: 2000, temperature: 0.7 } },
            payload: { message: { text: [{ role: 'user', content: prompt }] } }
        })
    });
    if (!response.ok) throw new Error(`Spark error: ${response.status}`);
    const data = await response.json();
    return data.payload.choices.text[0].content;
}

// ======================== 字节豆包 ========================
async function callDoubao(model, prompt) {
    const apiKey = process.env.DOUBAO_API_KEY;
    if (!apiKey) throw new Error('Missing Doubao API Key');
    const response = await fetch('https://ark.cn-beijing.volces.com/api/v3/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 2000,
            temperature: 0.7
        })
    });
    if (!response.ok) throw new Error(`Doubao error: ${response.status}`);
    const data = await response.json();
    return data.choices[0].message.content;
}

// ======================== 腾讯混元 ========================
async function callHunyuan(model, prompt) {
    const secretId = process.env.HUNYUAN_SECRET_ID;
    const secretKey = process.env.HUNYUAN_SECRET_KEY;
    if (!secretId || !secretKey) throw new Error('Missing Hunyuan credentials');
    // 此处需实现腾讯云签名，为简化，使用官方 SDK 或提供示例
    // 实际部署时建议使用官方 SDK 或简化调用
    // 这里提供一种简单 HTTP 调用方式（需获取临时密钥或使用 API 网关）
    // 推荐使用官方 SDK，本例仅作占位
    throw new Error('Hunyuan integration requires SDK setup. Please refer to Tencent Cloud documentation.');
}