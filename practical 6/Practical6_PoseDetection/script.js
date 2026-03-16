const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function setupCamera(){
  const stream = await navigator.mediaDevices.getUserMedia({
    video:true
  });

  video.srcObject = stream;
}

function drawKeypoints(keypoints){

  keypoints.forEach(point =>{

    if(point.score > 0.5){

      ctx.beginPath();
      ctx.arc(point.position.x, point.position.y,5,0,2*Math.PI);
      ctx.fillStyle="red";
      ctx.fill();

    }

  });

}

function drawSkeleton(keypoints){

  const adjacent = posenet.getAdjacentKeyPoints(keypoints,0.5);

  adjacent.forEach(pair =>{

    ctx.beginPath();

    ctx.moveTo(pair[0].position.x,pair[0].position.y);
    ctx.lineTo(pair[1].position.x,pair[1].position.y);

    ctx.strokeStyle="blue";
    ctx.lineWidth=2;
    ctx.stroke();

  });

}

async function detectPose(){

  const net = await posenet.load();

  async function poseDetectionFrame(){

    const pose = await net.estimateSinglePose(video);

    ctx.clearRect(0,0,600,500);
    ctx.drawImage(video,0,0,600,500);

    drawKeypoints(pose.keypoints);
    drawSkeleton(pose.keypoints);

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();

}

async function main(){

  await setupCamera();
  video.play();
  detectPose();

}

main();