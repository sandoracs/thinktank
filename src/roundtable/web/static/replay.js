/* Roundtable replay client (DESIGN.md §15 "Visszajátszás és export", M6).
 *
 * Steps through a finished session's event stream one event at a time. The full
 * event list is fetched once (the source of truth is the stored stream, so the
 * replay is identical to the live run — DESIGN.md §9.2).
 */
(function () {
  "use strict";
  var list = document.getElementById("rp-events");
  if (!list) return; // not the replay view

  var segments = location.pathname.split("/").filter(Boolean);
  var sid = segments[segments.length - 1];

  var events = [];
  var cursor = 0; // number of events currently revealed
  var playing = false;
  var timer = null;

  function describe(ev) {
    var t = ev.type;
    var p = ev.payload || {};
    if (t === "MessagePosted" && p.message) {
      return p.message.speaker_id + ": " + p.message.content;
    }
    if (t === "TurnAssigned") return "→ " + (p.speaker_id || "") + " szól (" + (p.strategy || "") + ")";
    if (t === "RoundStarted") return "Kör " + (p.round || "") + " kezdete";
    if (t === "RoundEnded") return "Kör " + (p.round || "") + " vége";
    if (t === "ReflectionProposed") return "Reflexiós javaslat (" + (p.agent_id || "") + ")";
    if (t === "PersonaUpdated") return "Perszóna frissítve (" + (p.agent_id || "") + ")";
    if (t === "ApprovalRequested") return "Jóváhagyásra vár (" + (p.agent_id || "") + ")";
    if (t === "ApprovalDecided") return "Jóváhagyás: " + (p.decision || "") + " (" + (p.agent_id || "") + ")";
    if (t === "SessionEnded") return "Session vége: " + (p.reason || "");
    if (t === "LLMCallCompleted")
      return "LLM hívás (" + (p.model || "") + ", $" + (p.cost_usd || 0) + ")";
    return t;
  }

  function render() {
    list.innerHTML = "";
    var shown = events.slice(0, cursor);
    for (var i = 0; i < shown.length; i++) {
      var ev = shown[i];
      var li = document.createElement("li");
      li.className = "rp-event";
      var head = document.createElement("span");
      head.className = "rp-seq";
      head.textContent = "#" + ev.seq + " " + ev.type;
      var body = document.createElement("div");
      body.className = "rp-body";
      body.textContent = describe(ev);
      li.appendChild(head);
      li.appendChild(body);
      list.appendChild(li);
    }
    document.getElementById("rp-progress").textContent = cursor + " / " + events.length + " esemény";
    if (shown.length) {
      var last = shown[shown.length - 1];
      if (last.type === "SessionEnded") {
        document.getElementById("rp-status").textContent = "ended";
      }
    }
    list.scrollTop = list.scrollHeight;
  }

  function stop() {
    playing = false;
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    var btn = document.getElementById("rp-play");
    if (btn) btn.textContent = "▶";
  }

  function tick() {
    if (cursor >= events.length) {
      stop();
      return;
    }
    cursor += 1;
    render();
    if (cursor < events.length) {
      timer = setTimeout(tick, 250);
    } else {
      stop();
    }
  }

  document.getElementById("rp-first").addEventListener("click", function () {
    stop();
    cursor = 0;
    render();
  });
  document.getElementById("rp-last").addEventListener("click", function () {
    stop();
    cursor = events.length;
    render();
  });
  document.getElementById("rp-prev").addEventListener("click", function () {
    stop();
    cursor = Math.max(0, cursor - 1);
    render();
  });
  document.getElementById("rp-next").addEventListener("click", function () {
    stop();
    cursor = Math.min(events.length, cursor + 1);
    render();
  });
  document.getElementById("rp-play").addEventListener("click", function () {
    if (playing) {
      stop();
      return;
    }
    if (cursor >= events.length) cursor = 0;
    playing = true;
    this.textContent = "⏸";
    tick();
  });

  fetch("/api/sessions/" + sid + "/events")
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (data) {
      events = data.slice().sort(function (a, b) {
        return a.seq - b.seq;
      });
      list.setAttribute("data-total", String(events.length));
      cursor = events.length; // start fully revealed; step back / replay from here
      render();
    })
    .catch(function (err) {
      document.getElementById("rp-progress").textContent = "Betöltés sikertelen: " + err.message;
    });
})();
