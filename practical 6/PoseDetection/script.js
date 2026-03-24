const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// 🎥 Camera Setup
async function setupCamera(){
  const stream = await navigator.mediaDevices.getUserMedia({ video:true });
  video.srcObject = stream;
}

// 🔴 Draw Keypoints
function drawKeypoints(keypoints){
  keypoints.forEach(point=>{
    if(point.score > 0.5){
      ctx.beginPath();
      ctx.arc(point.position.x, point.position.y,5,0,2*Math.PI);
      ctx.fillStyle="red";
      ctx.fill();
    }
  });
}

// 🔵 Draw Skeleton
function drawSkeleton(keypoints){
  const pairs = posenet.getAdjacentKeyPoints(keypoints,0.5);

  pairs.forEach(pair=>{
    ctx.beginPath();
    ctx.moveTo(pair[0].position.x, pair[0].position.y);
    ctx.lineTo(pair[1].position.x, pair[1].position.y);
    ctx.strokeStyle="blue";
    ctx.lineWidth=2;
    ctx.stroke();
  });
}

// 🧠 Better Sitting vs Standing Logic
function detectPosture(keypoints){

  const leftHip = keypoints.find(k=>k.part==="leftHip");
  const rightHip = keypoints.find(k=>k.part==="rightHip");
  const leftKnee = keypoints.find(k=>k.part==="leftKnee");
  const rightKnee = keypoints.find(k=>k.part==="rightKnee");

  if(leftHip.score>0.5 && rightHip.score>0.5 && leftKnee.score>0.5 && rightKnee.score>0.5){

    // Average positions
    const hipY = (leftHip.position.y + rightHip.position.y)/2;
    const kneeY = (leftKnee.position.y + rightKnee.position.y)/2;

    const diff = Math.abs(hipY - kneeY);

    // 🎯 Threshold Logic
    if(diff < 60){
      return { text:"🪑 Sitting", color:"yellow" };
    }else{
      return { text:"🧍 Standing", color:"lime" };
    }
  }

  return { text:"Detecting...", color:"white" };
}

// 🚀 Main Detection
async function detectPose(){

  const net = await posenet.load();

  async function poseFrame(){

    const pose = await net.estimateSinglePose(video);

    ctx.clearRect(0,0,600,500);
    ctx.drawImage(video,0,0,600,500);

    drawKeypoints(pose.keypoints);
    drawSkeleton(pose.keypoints);

    const result = detectPosture(pose.keypoints);

    // 🎯 Display Result
    ctx.font = "28px Arial";
    ctx.fillStyle = result.color;
    ctx.fillText(result.text, 20, 40);

    requestAnimationFrame(poseFrame);
  }

  poseFrame();
}

// ▶ Start
async function main(){
  await setupCamera();
  video.play();
  detectPose();
}

main();