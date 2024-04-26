import matplotlib.pyplot as plt
from PIL import Image
import os
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import imutils
import cv2
import torch
from .model import Net
import numpy as np
from torchvision import transforms
from sudoku import Sudoku

class solve_sudoku():
    def __init__(self, image_name, model=None):
        self.image= cv2.imread(image_name)
        self.thresh= self.threshold(self.image)
        if model is not None:
            self.model= model
        else:
            self.model= Net()
            self.model.load_state_dict(torch.load('checkpoint.pt'))
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(
                   (0.5,),(0.5,))
               ])
        
        (self.warped_image, self.cellLocs, self.solution, self.digits_per_cell)= self.solver()
        self.showsolution()


    def threshold(self, image= None):
        """Returns binary image with white foreground and black background"""
        if image is None:
            image= self.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        threshold_img = cv2.adaptiveThreshold(blurred, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
        threshold_img = cv2.bitwise_not(threshold_img)
        # plt.imshow(threshold_img, cmap='gray')

        return threshold_img
    
    def find_contours(self, thresh=None, image=None):
        """Finds the external contour in the image to get the outer lines of the sudoku box
           Performs four point transform to get the rectangular bird eye view of sudoku box only
           Returns the rectangular bird eye view of sudoku box only:
                warped_colred -> Colored bird eye view of sudoku box only
                warped_gray -> Gray bird eye view of sudoku box only
        """
        if image is None:
            image= self.image
        if thresh is None:
            thresh= self.thresh
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        puzzleCnt = None

        for c in cnts:
            line = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * line, True)

            if len(approx) == 4:
                puzzleCnt = approx
                break
        if puzzleCnt is None:
            raise Exception(("could not find sudoku puzzle outline."
                "try debugging your thresholding and contour steps."))
                
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0,255,0), 2)
        # plt.imshow(output)
        warped_colored = four_point_transform(image, puzzleCnt.reshape(4,2))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        warped_gray = four_point_transform(gray, puzzleCnt.reshape(4,2))
        return (warped_colored, warped_gray) 
    
    def display_sudoku_board(self, board):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)  # Print horizontal separator every 3 rows
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")  # Print vertical separator every 3 columns
                print(board[i][j], end=" ")
            
            print()

    def solver(self, image=None, thresh=None, model=None):
        if image is None:
            image= self.image
        if thresh is None:
            thresh= self.thresh
        if model is None:
            model= self.model
        image = imutils.resize(image, width = 600)
        thresh= imutils.resize(thresh, width=600)
        (warped_colored, warped_gray) = self.find_contours(thresh, image)
        board = np.zeros((9,9), dtype="int")
        stepX =  warped_gray.shape[1] // 9
        stepY =  warped_gray.shape[0] // 9
        cellLocs = []
        digits_per_cell=np.zeros((9,9)) #to track what cells have digits in it, used later to write the solution
        for y in range(0, 9):
            row = []
            for x in range(0, 9):
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1)*stepX
                endY = (y + 1)*stepY
                row.append((startX, startY, endX, endY))
                cell = warped_gray[startY:endY, startX:endX]
                digit = self._extract_digit(cell)
                if digit is not None:
                    digits_per_cell[y,x]=1
                    roi = cv2.resize(digit, (28, 28))
                    img = Image.fromarray(roi)
                    img = self.transform(img)
                    cell_value = img.to(self.device)
                    cell_value = cell_value.float()
                    cell_value = cell_value.unsqueeze(0)
                    model.eval()
                    out = model(cell_value.view(1, 1, 28, 28))
                    _, predict = torch.max(out, 1)
                    pred = predict.cpu().numpy()
                    board[y, x] = pred[0]
            cellLocs.append(row)
        print("Read the sudoku board:")
        self.display_sudoku_board(board)
        approval= input("Are the read values correct? Press N if you'd like to make changes: ")
        while approval.lower()=='n':
            try:
                row_to_correct= int(input('Enter the row of incorrect value (1-9): '))-1
                column_to_correct= int(input('Enter the column of incorrect value (1-9): '))-1
                value_to_replace= int(input('Enter the correct value (1-9): '))
                if not (0 <= row_to_correct < 9 and 0 <= column_to_correct < 9 and 1 <= value_to_replace <= 9):
                    print('\n')
                    raise ValueError("!!!Enter valid values for row, column, and value (1-9)!!!")
                
                board[row_to_correct, column_to_correct] = value_to_replace
                self.display_sudoku_board(board)
                approval = input("Are the updated values correct? Press 'N' to make changes: ")
            except ValueError:
                print("Enter only integer values for row, column, and value (1-9)")
            
        puzzle = Sudoku(3, 3, board=board.tolist())
        print("Solving sudoku puzzle...")
        solution = puzzle.solve()
        solution.show_full()
        return (warped_colored, cellLocs, solution, digits_per_cell)
    

    
    def _extract_digit(self, cell):
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            return None
        c = max(cnts, key = cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype=np.uint8) 
        cv2.drawContours(mask, [c], -1, 255, -1)
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w*h)
        if percentFilled < 0.03:
            return None
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)
        return digit
    


    def showsolution(self, warped_image=None, cellLocs=None, solution=None, digits_per_cell=None):
        if warped_image is None:
            warped_image= self.warped_image
        if cellLocs is None:
            cellLocs= self.cellLocs
        if solution is None:
            solution= self.solution
        if digits_per_cell is None:
            digits_per_cell= self.digits_per_cell
        for i,(cellRow, boardRow) in enumerate(zip(cellLocs, solution.board)):
            for j,(box, digit) in enumerate(zip(cellRow, boardRow)):
                startX, startY, endX, endY = box
                textX = int((endX - startX) * 0.33)
                textY = int((endY - startY) * -0.2)
                textX += startX
                textY += endY
                if digits_per_cell[i][j]==0:
                    cv2.putText(warped_image, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
        cv2.imshow('Solved', warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.imshow(warped_image)
        save_dir= "images/solved_puzzle.png"
        cv2.imwrite(save_dir, warped_image)
        print(f'Solved image saved at: {save_dir}')

if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        image= sys.argv[1]
        solve_sudoku(image)
    else:
        raise ValueError('Please provide image name as an argument')