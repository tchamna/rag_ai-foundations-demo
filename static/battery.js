const batteryWelcome = messages.innerHTML;
const originalReset = resetChat;
resetChat = function() { originalReset(); messages.innerHTML = batteryWelcome; messages.querySelectorAll('[data-prompt]').forEach(button => { button.onclick = () => { input.value = button.dataset.prompt; input.focus(); }; }); };
const originalAsk = ask;
ask = async function(message) { if (busy) return; messages.querySelector('.welcome')?.remove(); await originalAsk(message); };
const originalStatus = renderStatus;
renderStatus = function(status) { originalStatus(status); readyPill.innerHTML = '<span></span>Ready to learn'; };
