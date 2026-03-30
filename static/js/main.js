const speedInput = document.getElementById("speed");
const speedValue = document.getElementById("speed-value");
const laneColor = document.getElementById("lane-color");
const kpInput = document.getElementById("kp");
const kiInput = document.getElementById("ki");
const kdInput = document.getElementById("kd");
const modeChip = document.getElementById("mode-chip");
const yoloTag = document.getElementById("yolo-tag");
const aiMessage = document.getElementById("ai-message");
const laneOffset = document.getElementById("lane-offset");
const controlDebug = document.getElementById("control-debug");
const redLight = document.getElementById("red-light");
const objectList = document.getElementById("object-list");
const manualPanel = document.getElementById("manual-panel");
const manualBtn = document.getElementById("manual-mode-btn");
const autoBtn = document.getElementById("auto-mode-btn");
const settingsFeedback = document.getElementById("settings-feedback");

let currentMode = "manual";
let formDirty = false;
let isApplyingSettings = false;
let commandInFlight = false;
let lastSentAction = "stop";

function setModeUI(mode) {
    currentMode = mode;
    modeChip.textContent = mode.toUpperCase();
    manualBtn.classList.toggle("active", mode === "manual");
    autoBtn.classList.toggle("active", mode === "auto");
    manualPanel.classList.toggle("manual-disabled", mode !== "manual");
}

function updateSpeedLabel() {
    speedValue.textContent = Number(speedInput.value).toFixed(2);
}

function showSettingsFeedback(message, type) {
    settingsFeedback.textContent = message;
    settingsFeedback.className = `settings-feedback is-visible is-${type}`;
}

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    return response.json();
}

async function applySettings(modeOverride = null) {
    isApplyingSettings = true;
    showSettingsFeedback("套用設定中...", "pending");
    const payload = {
        mode: modeOverride || currentMode,
        lane_color: laneColor.value,
        speed: Number(speedInput.value),
        kp: Number(kpInput.value),
        ki: Number(kiInput.value),
        kd: Number(kdInput.value),
    };

    try {
        const result = await postJSON("/api/settings", payload);
        if (!result.ok) {
            showSettingsFeedback(result.error || "套用設定失敗", "error");
            return;
        }
        formDirty = false;
        syncState(result.state);
        showSettingsFeedback("套用成功", "success");
    } catch (error) {
        console.warn("apply settings failed", error);
        showSettingsFeedback("套用設定失敗", "error");
    } finally {
        isApplyingSettings = false;
    }
}

async function sendCommand(action) {
    if (currentMode !== "manual") {
        return;
    }
    if (commandInFlight) {
        return;
    }
    commandInFlight = true;
    try {
        const result = await postJSON("/control", { action });
        if (!result.ok) {
            showSettingsFeedback(result.error || "控制命令失敗", "error");
            return;
        }
        lastSentAction = action;
        if (action === "stop") {
            showSettingsFeedback("已停止", "success");
        } else {
            showSettingsFeedback(`手動控制: ${action}`, "success");
        }
        await refreshState();
    } catch (error) {
        console.warn("control failed", error);
        showSettingsFeedback("控制命令失敗", "error");
    } finally {
        commandInFlight = false;
    }
}

function handleDriveClick(action) {
    if (currentMode !== "manual") {
        return;
    }
    sendCommand(action);
}

function bindKeyboard() {
    const keyMap = {
        ArrowUp: "forward",
        KeyW: "forward",
        ArrowDown: "backward",
        KeyS: "backward",
        ArrowLeft: "left",
        KeyA: "left",
        ArrowRight: "right",
        KeyD: "right",
        Space: "stop",
    };

    document.addEventListener("keydown", (event) => {
        if (currentMode !== "manual" || event.repeat) {
            return;
        }
        const action = keyMap[event.code];
        if (!action) {
            return;
        }
        event.preventDefault();
        sendCommand(action);
    });
}

function syncState(state) {
    setModeUI(state.mode);
    if (!formDirty && !isApplyingSettings) {
        speedInput.value = state.speed;
        updateSpeedLabel();
        laneColor.value = state.lane_color;
        kpInput.value = state.pid.kp;
        kiInput.value = state.pid.ki;
        kdInput.value = state.pid.kd;
    }
    aiMessage.textContent = state.ai_message;
    laneOffset.textContent = Number(state.lane_offset).toFixed(1);
    controlDebug.textContent = state.control_debug || "idle";
    redLight.textContent = state.red_light ? "RED LIGHT" : "CLEAR";
    redLight.className = state.red_light ? "red-light" : "safe-light";
    objectList.textContent = state.objects.length ? state.objects.join(", ") : "No objects";
    yoloTag.textContent = state.yolo_enabled ? "YOLO enabled" : "YOLO model missing";

    if (!state.motor_enabled) {
        showSettingsFeedback(`馬達不可用: ${state.motor_error || "unknown error"}`, "error");
    }
}

async function refreshState() {
    try {
        const response = await fetch("/api/state");
        const state = await response.json();
        syncState(state);
    } catch (error) {
        console.warn("state refresh failed", error);
    }
}

function markFormDirty() {
    formDirty = true;
}

speedInput.addEventListener("input", () => {
    updateSpeedLabel();
    markFormDirty();
});
laneColor.addEventListener("change", markFormDirty);
kpInput.addEventListener("input", markFormDirty);
kiInput.addEventListener("input", markFormDirty);
kdInput.addEventListener("input", markFormDirty);
document.getElementById("apply-settings").addEventListener("click", () => applySettings());
manualBtn.addEventListener("click", () => applySettings("manual"));
autoBtn.addEventListener("click", () => applySettings("auto"));

window.handleDriveClick = handleDriveClick;

bindKeyboard();
updateSpeedLabel();
refreshState();
setInterval(refreshState, 700);
