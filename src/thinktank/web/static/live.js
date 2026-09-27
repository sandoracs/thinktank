/* ThinkTank live-table client (DESIGN.md §14.2, §15).
 *
 * One WebSocket per session. The server sends, in order:
 *   hello  -> {"type":"hello","last_seq":N,...}
 *   events -> {"type":"event","seq":S,"event":{...},"fragments":{transcript,sidebar}}
 *   your_turn -> {"type":"your_turn","participant":pid,"deadline":iso}   (humans only)
 * The client appends the server-rendered transcript fragment and swaps the
 * sidebar; it never parses message content itself.
 */
(function () {
  "use strict";
  var transcriptEl = document.getElementById("transcript");
  if (!transcriptEl) return; // not the live view

  var params = new URLSearchParams(location.search);
  var segments = location.pathname.split("/").filter(Boolean);
  var sid = segments[segments.length - 1];
  var humanParticipant = params.get("participant");
  var humanToken = params.get("token") || "";

  var lastSeq = parseInt(transcriptEl.getAttribute("data-last-seq") || "0", 10);
  var status = "unknown";
  var myTurn = false;
  var ended = false;
  var ws = null;
  var reconnectTimer = null;

  function wsUrl() {
    var proto = location.protocol === "https:" ? "wss" : "ws";
    var url = proto + "://" + location.host + "/ws/sessions/" + encodeURIComponent(sid) + "?after_seq=" + lastSeq;
    if (humanParticipant) {
      url += "&participant=" + encodeURIComponent(humanParticipant) + "&token=" + encodeURIComponent(humanToken);
    }
    return url;
  }

  function appendFragment(frag) {
    if (!frag) return;
    var holder = document.createElement("div");
    holder.innerHTML = frag;
    while (holder.firstChild) transcriptEl.appendChild(holder.firstChild);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function setSidebar(html) {
    if (!html) return;
    var old = document.querySelector(".sidebar");
    var holder = document.createElement("div");
    holder.innerHTML = html;
    var fresh = holder.querySelector(".sidebar");
    if (old && fresh) old.replaceWith(fresh);
  }

  function setMyTurn(on) {
    myTurn = on;
    var hint = document.getElementById("turn-hint");
    if (hint) {
      hint.textContent = on ? "Your turn!" : "";
      hint.classList.toggle("active", on);
    }
    syncControls();
  }

  function syncControls() {
    var setBtn = function (name, disabled) {
      var el = document.querySelector('[data-ctl="' + name + '"]');
      if (el) el.disabled = disabled;
    };
    var live = status === "running" || status === "paused";
    setBtn("pause", status !== "running");
    setBtn("resume", status !== "paused");
    setBtn("stop", !live);
    setBtn("start", status !== "created");
    setBtn("reset", status === "unknown");
    var edit = document.getElementById("edit-ctl");
    if (edit) edit.setAttribute("aria-disabled", status === "created" ? "false" : "true");
    var del = document.querySelector('[data-ctl="delete"]');
    if (del) del.disabled = !(status === "created" || status === "ended" || status === "interrupted");

    var input = document.getElementById("say-input");
    var send = document.getElementById("say-send");
    var hand = document.getElementById("raise-hand");
    var canTalk = !!humanParticipant && status === "running" && myTurn;
    if (input) input.disabled = !canTalk;
    if (send) send.disabled = !canTalk;
    if (hand) hand.disabled = !(!!humanParticipant && status === "running");
  }

  function applyEvent(ev) {
    var type = ev.type;
    var payload = ev.payload || {};
    if (type === "SessionStarted") status = "running";
    else if (type === "SessionPaused") status = "paused";
    else if (type === "SessionResumed") status = "running";
    else if (type === "SessionEnded") {
      status = "ended";
      ended = true;
    }
    if (type === "TurnAssigned" && payload.speaker_id === humanParticipant) setMyTurn(true);
    var speaker = payload.speaker_id || (payload.message && payload.message.speaker_id);
    if ((type === "MessagePosted" || type === "TurnSkipped") && speaker === humanParticipant) {
      setMyTurn(false);
    }
    syncControls();
  }

  function onMessage(raw) {
    var msg;
    try {
      msg = JSON.parse(raw);
    } catch (e) {
      return;
    }
    if (msg.type === "hello") {
      if (msg.status) status = msg.status;
      syncControls();
      return;
    }
    if (msg.type === "your_turn") {
      if (msg.participant === humanParticipant) setMyTurn(true);
      return;
    }
    if (msg.type === "event") {
      if (msg.seq <= lastSeq) return; // already rendered (replay overlap)
      lastSeq = msg.seq;
      transcriptEl.setAttribute("data-last-seq", String(lastSeq));
      appendFragment(msg.fragments && msg.fragments.transcript);
      setSidebar(msg.fragments && msg.fragments.sidebar);
      applyEvent(msg.event || {});
    }
  }

  function connect() {
    if (ws) return;
    ws = new WebSocket(wsUrl());
    ws.onmessage = function (e) {
      onMessage(e.data);
    };
    ws.onclose = function () {
      ws = null;
      if (!ended) {
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connect, 1500); // lossless reconnect: after_seq carries state
      }
    };
    ws.onerror = function () {
      try {
        ws.close();
      } catch (e) {
        /* already closed */
      }
    };
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  var input = document.getElementById("say-input");
  var sendBtn = document.getElementById("say-send");
  var handBtn = document.getElementById("raise-hand");

  function doSay() {
    if (!input) return;
    var content = input.value.trim();
    if (!content) return;
    send({ type: "say", content: content });
    input.value = "";
  }

  if (sendBtn) sendBtn.addEventListener("click", doSay);
  if (input) {
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        doSay();
      }
    });
  }
  if (handBtn) handBtn.addEventListener("click", function () {
    send({ type: "raise_hand" });
  });

  // Approval panel (DESIGN.md §12.3, M6): decide a pending persona change.
  var approvalPanel = document.getElementById("approvals-panel");
  if (approvalPanel) {
    approvalPanel.addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-decision]");
      if (!btn) return;
      var card = btn.closest(".approval");
      var agent = card.getAttribute("data-agent");
      var decision = btn.getAttribute("data-decision");
      btn.disabled = true;
      fetch("/api/sessions/" + sid + "/approvals/" + encodeURIComponent(agent), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision: decision }),
      })
        .then(function (r) {
          if (!r.ok) throw new Error(r.status);
          location.reload();
        })
        .catch(function (err) {
          btn.disabled = false;
          alert("Decision failed: " + err.message);
        });
    });
  }

  // Control bar: start / pause / resume / stop / reset (DESIGN.md §15).
  var controls = document.querySelector(".controls");
  if (controls) {
    controls.addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-ctl]");
      if (!btn || btn.disabled) return;
      var name = btn.getAttribute("data-ctl");
      if (name === "reset" && !confirm("Really reset to the beginning? The conversation and the events will be deleted.")) return;
      if (name === "delete" && !confirm("Really delete this session? The conversation and the events will be permanently deleted.")) return;
      btn.disabled = true;
      var origLabel = btn.textContent;
      if (name === "reset") {
        // The server waits for the live engine to finish its current turn
        // before wiping, which can take a few seconds with a real LLM.
        btn.textContent = "Resetting…";
      }
      if (name === "delete") {
        btn.textContent = "Deleting…";
        fetch("/api/sessions/" + sid, { method: "DELETE" })
          .then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            location.href = "/";
          })
          .catch(function (err) {
            btn.disabled = false;
            btn.textContent = origLabel;
            alert("Deletion failed: " + err.message);
          });
        return;
      }
      fetch("/api/sessions/" + sid + "/" + name, { method: "POST" })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          location.reload();
        })
        .catch(function (err) {
          btn.disabled = false;
          btn.textContent = origLabel;
          alert(name + " failed: " + err.message);
        });
    });
  }

  connect();
  syncControls();
})();
