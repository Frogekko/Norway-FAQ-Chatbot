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

document.getElementById("train-but").addEventListener("click", async () => {
  const iterationsInput = document.getElementById("train-iterations");
  const iterations = parseInt(iterationsInput.value, 10) || 1000;
  const btn = document.getElementById("train-but");
  const logBox = document.getElementById("training-log");

  btn.disabled = true;
  btn.textContent = "Training...";
  logBox.textContent = "";

  try {
    const response = await fetch("/api/train-more", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ iterations: 1000, Batch_Size: 64 })
    });

    if (!response.ok) throw new Error("Failed to start training");

    const evtSource = new EventSource("/api/train-stream");

    evtSource.onmessage = (event) => {
      console.log("Recieved Training Message:", event.data);
      logBox.textContent += `${event.data}\n`;
      logBox.scrollTop = logBox.scrollHeight;

      if (event.data.includes("[Training] Done")) {
        btn.disabled = false;
        btn.textContent = "Start Training";
        evtSource.close();
      }
    };

    evtSource.onerror = (error) => {
      console.error("EventSource error:", error);
      logBox.textContent += "\n[Error] Connection lost to training stream.";
      btn.disabled = false;
      btn.textContent = "Start Training";
      evtSource.close();
    };

  } catch (error) {
    console.error("Training initiation error:", error);
    logBox.textContent += "[Error] Could not start training.";
    btn.disabled = false;
    btn.textContent = "Start Training";
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