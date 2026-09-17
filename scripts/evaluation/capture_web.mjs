// Local-only browser QA. Does not submit playback or human feedback.
import fs from 'node:fs/promises';
import path from 'node:path';
const output = path.resolve('results/retraining_20260911/web_qa');
await fs.mkdir(output, {recursive:true});
const tab = await (await fetch('http://127.0.0.1:9223/json/new?about:blank', {method:'PUT'})).json();
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((resolve,reject) => {ws.onopen=resolve;ws.onerror=reject;});
let seq=0;const pending=new Map();
ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
async function until(expression,seconds=120){const start=Date.now();while(Date.now()-start<seconds*1000){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,500));}throw Error('Timed out: '+expression);}
async function capture(name){const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(output,name+'.png'),Buffer.from(shot.data,'base64'));}
await call('Page.enable');await call('Runtime.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:'http://127.0.0.1:5173/'});
await until('document.querySelector(".primary-button") !== null',30);
await capture('first_page');
await evaluate('Array.from(document.querySelectorAll("button")).find(b=>b.textContent.includes("第二篇")).click()');
await until('document.querySelector("textarea").placeholder.includes("我现在很烦")',10);
await evaluate('(()=>{const t=document.querySelector("textarea");Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,"value").set.call(t,"我最近有点低落，想开心起来，喜欢吉他音乐。");t.dispatchEvent(new Event("input",{bubbles:true}));document.querySelector(".primary-button").click();})()');
await until('!document.querySelector(".primary-button").disabled',180);
const error=await evaluate('document.querySelector(".error")?.textContent || ""');
if(error)throw Error(error);
await until('document.body.innerText.includes("retraining_20260911")',30);
await capture('second_page');
const qa=await evaluate('({title:document.title,text:document.body.innerText,overflow:document.documentElement.scrollWidth>innerWidth})');
await fs.writeFile(path.join(output,'browser_qa.json'),JSON.stringify({status:'passed',...qa,real_feedback_submitted:false},null,2));
console.log(JSON.stringify({status:'passed',output,horizontal_overflow:qa.overflow}));
ws.close();
