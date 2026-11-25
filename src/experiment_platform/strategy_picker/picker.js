(() => {
  'use strict';
  const chart = document.getElementById('chart');
  const feedback = document.getElementById('feedback');
  let args = null, initialized = false, drawing = false, busy = false;
  let pending = null, sequence = 0, lastFigure = '', queue = Promise.resolve();
  let linePress=null;
  const post = (type, data) => window.parent.postMessage({isStreamlitMessage:true, type, ...data}, '*');
  const nearest = x => (args?.choices || []).reduce((best, choice) => !best || Math.abs(choice.value-x)<Math.abs(best.value-x) ? choice : best, null);
  const height = () => post('streamlit:setFrameHeight',{height:Math.ceil(document.body.getBoundingClientRect().height)});
  const status = text => {feedback.textContent = text;height();};
  const protectFixedShapes = () => {
    chart.querySelectorAll('.shapelayer [data-index]').forEach(element=>{
      if(!Object.prototype.hasOwnProperty.call(args?.draggable||{},element.getAttribute('data-index'))) {
        element.style.pointerEvents='none';
        element.querySelectorAll('*').forEach(child=>{child.style.pointerEvents='none';});
      }
    });
  };
  const send = (action, label, previous) => {
    if (busy || !args || !label) return;
    busy = true;
    pending = Date.now().toString(36)+'-'+(++sequence)+'-'+Math.random().toString(36).slice(2,9);
    status('Updating Tests…');
    post('streamlit:setComponentValue', {dataType:'json',value:{id:pending,revision:args.revision,action,strategy:label,previous}});
  };
  const clickPoint = event => {
    if (drawing || busy || !args || !event.points?.length) return;
    const x = Number(event.points[0].x);
    if (!Number.isFinite(x)) return;
    if (args.control_value !== null && Math.abs(x-args.control_value)<1e-8) {
      status('Control stays fixed. Change it in the settings.'); return;
    }
    const choice = nearest(x);
    if (choice && Math.abs(choice.value-x)<1e-8) send('toggle',choice.label);
    else status('This credit line is not available for all selected Partners and metrics. Choose another data point.');
  };
  const moveShape = event => {
    if (drawing || busy || !args) return;
    for (const [index, previous] of Object.entries(args.draggable || {})) {
      const prefix = 'shapes['+index+']';
      const update = event[prefix];
      const x0 = event[prefix+'.x0'] ?? update?.x0;
      const x1 = event[prefix+'.x1'] ?? update?.x1;
      if (!Number.isFinite(x0) && !Number.isFinite(x1)) continue;
      const x = Number.isFinite(x0) && Number.isFinite(x1) ? (x0+x1)/2 : (x0 ?? x1);
      const target = nearest(x);
      if (target) {
        drawing=true;
        Plotly.relayout(chart,{[prefix+'.x0']:target.value,[prefix+'.x1']:target.value,[prefix+'.y0']:0,[prefix+'.y1']:1})
          .then(()=>{drawing=false;send('move',target.label,previous);})
          .catch(()=>{drawing=false;status('Use the Test options on the right to move this line.');});
      }
      return;
    }
  };
  const render = async incoming => {
    args = incoming;
    const config = {responsive:true,displayModeBar:false,scrollZoom:false,
      editable:false,edits:{shapePosition:true},doubleClick:false};
    const figure = JSON.parse(args.figure);
    if (args.figure !== lastFigure) {
      drawing = true;
      try {
        if (!initialized) {
          await Plotly.newPlot(chart,figure.data,figure.layout,config);
          chart.on('plotly_click',clickPoint);
          chart.on('plotly_relayout',moveShape);
          chart.on('plotly_afterplot',protectFixedShapes);
          initialized = true;
        } else await Plotly.react(chart,figure.data,figure.layout,config);
        protectFixedShapes();
        lastFigure = args.figure;
      } finally { drawing = false; }
    }
    if (!pending || args.ack === pending) {
      busy = false; pending = null;
      status(args.message || 'Click a data point to toggle a Test. Drag a Test line to move it.');
    }
    height();
  };
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    const incoming = event.data.args;
    queue = queue.then(()=>render(incoming)).catch(error=>{
      busy=false;pending=null;drawing=false;
      status('Chart unavailable. Use the Test options on the right.');
      console.error('Strategy picker:',error);
      height();
    });
  });
  chart.addEventListener('pointerdown',event=>{
    if(busy||drawing)return;
    const shape=event.target.closest?.('[data-index]');
    const index=shape?.getAttribute('data-index');
    if(index!==null&&args?.draggable?.[index])linePress={x:event.clientX,y:event.clientY,label:args.draggable[index]};
  },true);
  window.addEventListener('pointerup',event=>{
    const press=linePress;linePress=null;
    if(press&&Math.hypot(event.clientX-press.x,event.clientY-press.y)<4)send('toggle',press.label);
  },true);
  let resizeFrame,lastWidth=0;
  new ResizeObserver(()=>{
    const width=document.body.getBoundingClientRect().width;
    if(width===lastWidth)return;
    lastWidth=width;
    cancelAnimationFrame(resizeFrame);
    resizeFrame=requestAnimationFrame(()=>{if(initialized)Plotly.Plots.resize(chart);});
  }).observe(document.body);
  new IntersectionObserver(entries=>{
    if(initialized&&!busy&&entries.some(entry=>entry.isIntersecting)) {
      drawing=true;
      Plotly.redraw(chart).then(()=>{drawing=false;protectFixedShapes();height();}).catch(()=>{drawing=false;});
    }
  }).observe(chart);
  post('streamlit:componentReady',{apiVersion:1});
})();
