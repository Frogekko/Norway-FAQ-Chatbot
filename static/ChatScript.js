document.getElementById("chat-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const input = document.getElementById("user-input");
  const maxLengthInput = document.getElementById("max-length-input");

  const message = input.value.trim();
  const max_length = parseInt(maxLengthInput.value) || 20;

  if (!message) return;

  addMessage("user", message);
  input.value = "";

  // Show typing placeholder
  addMessage("bot", bot_name + " is typing...");
  const loadingMsg = document.querySelector("#chat-box .message.bot:last-child");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, max_length })
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error: ${res.status}`);
    }

    const data = await res.json();
    console.log("Response data:", data);

    // Update text response
    loadingMsg.textContent = `${data.bot_name}: ${data.response.Response || "Error"}`;

    // Handle audio if present
    if (data.response.audio_base64) {
      const audioBlob = base64ToBlob(data.response.audio_base64, 'audio/mp3');
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play().catch(err => {
        console.error("Audio playback failed:", err);
        addMessage("bot", "Sorry, audio playback failed.");
      });
    } else {
      console.warn("No audio data received.");
      addMessage("bot", "Audio unavailable for this response.");
    }
  } catch (err) {
    console.error("Fetch error:", err);
    loadingMsg.textContent = `Error: ${err.message || "Server error. Is the backend running?"}`;
  }
});

function addMessage(sender, text) {
  const chatBox = document.getElementById("chat-box");
  const msg = document.createElement("div");
  msg.className = "message " + sender;
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function base64ToBlob(base64, mimeType) {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}