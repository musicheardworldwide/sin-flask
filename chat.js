const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatLog = document.getElementById('chat-log');

sendBtn.addEventListener('click', async () => {
  const userInputValue = userInput.value.trim();
  if (userInputValue !== '') {
    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: userInputValue }),
      });
      const responseJson = await response.json();
      chatLog.innerHTML += `<p>You: ${userInputValue}</p>`;
      chatLog.innerHTML += `<p>Ollama: ${responseJson.output}</p>`;
      userInput.value = '';
    } catch (error) {
      console.error(error);
    }
  }
});
