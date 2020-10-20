pragma solidity ^0.5.0;

contract StoreHash {
 string ipfsHash;

 function sendHash(string memory x) public {
   ipfsHash = x;
 }

 function getHash() public view returns (string memory x) {
   return ipfsHash;
 }
}
