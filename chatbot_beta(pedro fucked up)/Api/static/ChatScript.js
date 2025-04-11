document.getElementById("chat-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const input = document.getElementById("user-input"); // reads the user input file
  const maxLengthInput = document.getElementById("max-length-input"); // initializes the max length of the input

  const message = input.value.trim();
  const max_length = parseInt(maxLengthInput.value) || 20; // default 20 if empty

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
      body: JSON.stringify({ message, max_length }) // Send max_length dynamically
    });
    const data = await res.json();
    console.log(data);

    loadingMsg.textContent = `${data.bot_name}: ${data.response.Response || "Error"}`;
  } catch (err) {
    loadingMsg.textContent = "Server error. Is the backend running?";
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